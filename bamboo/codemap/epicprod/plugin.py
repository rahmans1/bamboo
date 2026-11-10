"""swf-epicprod Code Map plugin, built on the shared ORM recognizers (:mod:`bamboo.codemap.orm`).

swf-epicprod is a Django project, so where PanDA's recognizers read SQL text
and infer the row's class from how it is used, most writes here go through
the ORM and *name* their class.  The ORM part -- models, write forms,
attribution, values and conditions -- is :mod:`bamboo.codemap.orm`'s; this
plugin adds what is specific to swf-epicprod:

* which packages are read (:data:`PACKAGES`),
* raw ``UPDATE <table> SET <col> = ...`` text, mapped to a model by its
  ``db_table`` -- ``certain``,
* calls into PanDA (``pandaclient`` ``Client.*`` and ``requests`` to
  ``PANDA_URL``) as outbound boundaries, so a PanDA map can bind to them
  instead of a link being invented.

Values are read inside ``sql.across_calls`` over the whole corpus, so a
constant another module declares (``PROPAGATION_STATES`` imported from
``pcs.models``) settles the way a local one does.

Not done yet: the read side as statements (``ReadSiteNode``), log sites,
reach graph and entry points, and writes through ``save(update_fields=...)``
helpers that take the field name as data.

Scope: ``pcs/``, ``swf_epicprod/`` and ``tools/`` are read; ``scripts/``,
``pilot/``, migrations and tests are not, so PanDA use from those is not in
the map.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.calls import Calls, Types
from bamboo.codemap.gitsource import blob_sha, describe
from bamboo.codemap.models import BoundaryNode, MapFragment
from bamboo.codemap.orm.assemble import FieldWrite, assemble, write_condition
from bamboo.codemap.orm.resolve import Resolved, resolve, settler
from bamboo.codemap.orm.schema import DjangoModel, dotted, module_constants, read_models
from bamboo.codemap.orm.writes import scan_function
from bamboo.codemap.panda import pathcond, sql

logger = logging.getLogger(__name__)

MAP_ID = "swf-epicprod"

#: Top-level packages analysed, in the order they are walked.
PACKAGES = ("pcs", "swf_epicprod", "tools")

_SQL_UPDATE = re.compile(
    r"\bUPDATE\s+\"?(\w+)\"?\s+(?:AS\s+\w+\s+|\w+\s+)?SET\s+(.*?)(?:\bWHERE\b|\bFROM\b|\bRETURNING\b|$)",
    re.I | re.S,
)
_SQL_ASSIGN = re.compile(
    r"(?:^|,)\s*\"?(\w+)\"?\s*=\s*('(?:[^']*)'|%s|%\(\w+\)s|:\w+|[\w.]+)", re.S
)


def _own_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """Every node of *func*'s body, not those of functions or classes nested in it."""
    stack = list(reversed(getattr(func, "body", [])))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


class EpicprodCodeMapPlugin(CodeMapPlugin):
    """Builds the swf-epicprod fragment from a checkout."""

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._stamp = ""
        self.models: dict[str, DjangoModel] = {}
        self.writes: list[FieldWrite] = []
        self.selected: dict[tuple[str, str], set[str]] = {}
        self.boundaries: dict[str, BoundaryNode] = {}
        self.unresolved_writes = 0

    @property
    def map_id(self) -> str:
        return MAP_ID

    def prepare(self, source_root=None, schema_root=None, instance_record=None) -> str:
        if source_root is None:
            raise FileNotFoundError(
                "swf-epicprod is not an installed distribution: pass --source-root <checkout>"
            )
        root = Path(source_root).resolve()
        if not any((root / pkg).is_dir() for pkg in PACKAGES):
            raise FileNotFoundError(f"{root} holds none of {PACKAGES}")
        self._root = root
        self._stamp = f"swf-epicprod {describe(root) or 'unversioned'}"
        return self._stamp

    # ----------------------------------------------------------------- sources

    def _sources(self) -> Iterator[tuple[str, str, ast.Module, str]]:
        assert self._root is not None
        for pkg in PACKAGES:
            base = self._root / pkg
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.py")):
                rel = path.relative_to(self._root).as_posix()
                parts = rel.split("/")
                if (
                    "migrations" in parts
                    or "tests" in parts
                    or parts[-1].startswith("test")
                ):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(text, filename=rel)
                except SyntaxError as exc:
                    logger.warning("skipping %s: %s", rel, exc)
                    continue
                module = rel[:-3].replace("/", ".")
                if module.endswith(".__init__"):
                    module = module[: -len(".__init__")]
                yield rel, module, tree, text

    # ----------------------------------------------------------------- one function

    def _scan_function(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        qual: str,
        cls: Optional[str],
        rel: str,
        constants: dict[str, list[str]],
        panda_aliases: set[str],
    ) -> None:
        owner = f"{rel}::{qual}"
        settle = settler(self.models, constants)
        scan = scan_function(func, cls, self.models, constants)
        for key, vals in scan.selected.items():
            self.selected.setdefault(key, set()).update(vals)
        for w in scan.writes:
            self.writes.append(
                FieldWrite(
                    model=w.model,
                    field=w.field,
                    owner=owner,
                    rel_path=rel,
                    line=w.line,
                    form=w.form,
                    attribution=w.attribution,
                    arms=resolve(
                        w.value, w.at, func, settle, self.models, model=w.model
                    ),
                    path_condition=write_condition(w.at),
                    structural=w.structural,
                )
            )
        for node in _own_nodes(func):
            if isinstance(node, ast.Call):
                self._visit_boundary(node, owner, rel, panda_aliases)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._visit_sql(node.value, node, owner, rel)
            elif isinstance(node, ast.JoinedStr):
                text = "".join(
                    v.value if isinstance(v, ast.Constant) else "%s"
                    for v in node.values
                )
                self._visit_sql(text, node, owner, rel)

    def _visit_sql(self, text: str, at: ast.AST, owner: str, rel: str) -> None:
        for m in _SQL_UPDATE.finditer(text):
            table, sets = m.group(1), m.group(2)
            model = next(
                (x.name for x in self.models.values() if x.db_table == table), None
            )
            if model is None:
                continue
            for col, raw_value in _SQL_ASSIGN.findall(sets):
                if col not in self.models[model].state_fields():
                    continue
                arm = (
                    Resolved(raw_value.strip("'"), 1, [])
                    if raw_value.startswith("'")
                    else Resolved(f"runtime({raw_value})", 2, [])
                )
                self.writes.append(
                    FieldWrite(
                        model=model,
                        field=col,
                        owner=owner,
                        rel_path=rel,
                        line=at.lineno,  # type: ignore[attr-defined]
                        form="sql.update",
                        attribution="certain",
                        arms=[arm],
                        path_condition=write_condition(at),
                    )
                )

    def _visit_boundary(
        self, call: ast.Call, owner: str, rel: str, panda_aliases: set[str]
    ) -> None:
        name = dotted(call.func) or ""
        head = name.split(".")[0]
        if head in panda_aliases or name.startswith("pandaclient."):
            method = name.split(".")[-1]
            if not method.startswith(
                "_"
            ):  # the client's own plumbing is not a PanDA interface
                self._boundary(
                    owner,
                    f"pandaclient.Client.{method}",
                    "pandaclient",
                    rel,
                    call.lineno,
                )
        elif name in ("requests.get", "requests.post", "requests.put") and call.args:
            url = call.args[0]
            if isinstance(url, ast.JoinedStr) and any(
                isinstance(v, ast.FormattedValue) and "PANDA" in ast.unparse(v.value)
                for v in url.values
            ):
                path = "".join(
                    v.value for v in url.values if isinstance(v, ast.Constant)
                )
                self._boundary(
                    owner,
                    path or "<panda url>",
                    "http",
                    rel,
                    call.lineno,
                    verb=name.split(".")[-1].upper(),
                )

    def _boundary(
        self,
        owner: str,
        calls_out: str,
        transport: str,
        rel: str,
        line: int,
        verb: str = "",
    ) -> None:
        name = f"{MAP_ID}->panda::{calls_out}"
        node = self.boundaries.get(name)
        if node is None:
            node = BoundaryNode(
                name=name,
                map_id=MAP_ID,
                derived_from=self._stamp,
                system="panda",
                kind="transports_causation",
                transport="http",
                interface=calls_out,
                direction="outbound",
                metadata={"client": transport},
            )
            self.boundaries[name] = node
        if owner not in node.calls_out:
            node.calls_out.append(owner)
        node.call_sites.append(f"{rel}:{line}")
        if verb and verb not in node.verbs:
            node.verbs.append(verb)

    # ----------------------------------------------------------------- run

    def run(self) -> MapFragment:
        if self._root is None:
            raise RuntimeError("prepare() must be called before run()")
        parsed = list(self._sources())
        shas = {rel: blob_sha(text) for rel, _m, _t, text in parsed}
        for _rel, _m, tree, _text in parsed:
            pathcond.attach_parents(tree)
        self.models = read_models(
            (rel, module, tree) for rel, module, tree, _t in parsed
        )
        roots = {
            pkg: self._root / pkg for pkg in PACKAGES if (self._root / pkg).is_dir()
        }
        calls = Calls(
            Types(roots, {rel: (shas[rel], tree) for rel, _m, tree, _t in parsed})
        )

        with sql.across_calls(calls):
            for rel, _module, tree, _text in parsed:
                constants = module_constants(tree)
                panda_aliases: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            if a.name.startswith("pandaclient"):
                                panda_aliases.add(a.asname or a.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom) and (
                        node.module or ""
                    ).startswith("pandaclient"):
                        panda_aliases.update(a.asname or a.name for a in node.names)

                def functions(body, prefix: str, cls: Optional[str]):
                    for stmt in body:
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            qual = f"{prefix}{stmt.name}"
                            yield stmt, qual, cls
                            yield from functions(stmt.body, f"{qual}.", cls)
                        elif isinstance(stmt, ast.ClassDef):
                            yield from functions(
                                stmt.body, f"{prefix}{stmt.name}.", stmt.name
                            )

                for func, qual, cls in functions(tree.body, "", None):
                    self._scan_function(func, qual, cls, rel, constants, panda_aliases)

        subjects, entities, junctions, self.unresolved_writes = assemble(
            self.writes, self.models, MAP_ID, self._stamp, shas, self.selected
        )
        return MapFragment(
            map_id=MAP_ID,
            derived_from=self._stamp,
            subjects=subjects,
            entities=entities,
            junctions=junctions,
            boundaries=sorted(self.boundaries.values(), key=lambda b: b.name),
        )
