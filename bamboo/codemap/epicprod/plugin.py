"""swf-epicprod Code Map plugin -- a first, deliberately small one.

swf-epicprod is a Django project, so where PanDA's recognizers read SQL text
and infer the row's class from how it is used, most writes here go through
the ORM and *name* their class: ``ProdTask.objects.filter(...).update(status=
'submitted')``.  The declared models give the subjects and their vocabulary
(``choices=``) directly, which is what this plugin leans on.

Write forms recognised:

* ``obj.<field> = value`` -- attributed ``certain`` where the function binds
  ``obj`` from ``Model.objects...get()/create()/first()``, ``Model(...)``, a
  loop over a ``Model.objects`` queryset, an annotation, or ``self`` inside the
  model; ``structural`` where exactly one model declares every field touched
  on ``obj``; ``heuristic`` where the variable's name ends a model's name;
  else ``unresolved``.
* ``Model.objects....update(f=v)`` / ``.create(f=v)`` / ``update_or_create`` /
  ``get_or_create`` (kwargs and ``defaults=``), also through a queryset
  variable -- ``certain``.
* Raw ``UPDATE <table> SET <col> = ...`` text, mapped to a model by its
  ``db_table`` -- ``certain``.

Only *state-like* fields become subjects: a field with ``choices``, or named
``status``, ``state`` or ``*_status``.  Calls into PanDA (``pandaclient``
``Client.*`` and ``requests`` to ``PANDA_URL``) become outbound boundaries, so a
later PanDA map can bind to them instead of a link being invented.

Not done yet (the gap list for the spike): the read side as statements
(``ReadSiteNode``), log sites, reach graph/entry points, value enums, and
writes through ``save(update_fields=...)`` helpers that take the field name as
data.

Scope: only ``pcs/`` and ``swf_epicprod/`` are read (:data:`PACKAGES`);
``scripts/``, ``tools/``, ``pilot/``, migrations and tests are not.  So the
boundaries found are not all of swf-epicprod's PanDA use: task submission
(``Client.insertTaskParams``) appears under ``tools/`` and is not in the map.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.gitsource import blob_sha, describe
from bamboo.codemap.models import (
    Anchor,
    BoundaryNode,
    Branch,
    EntityNode,
    JunctionNode,
    MapFragment,
    SubjectNode,
)

logger = logging.getLogger(__name__)

MAP_ID = "swf-epicprod"

#: Top-level packages analysed, in the order they are walked.
PACKAGES = ("pcs", "swf_epicprod")

_SINGLE_OBJECT = {
    "get",
    "create",
    "first",
    "last",
    "latest",
    "earliest",
    "get_object_or_404",
}
_ORM_WRITE = {"update", "create", "update_or_create", "get_or_create"}
_ORM_SELECT = {"filter", "exclude", "get"}
_SQL_UPDATE = re.compile(
    r"\bUPDATE\s+\"?(\w+)\"?\s+(?:AS\s+\w+\s+|\w+\s+)?SET\s+(.*?)(?:\bWHERE\b|\bFROM\b|\bRETURNING\b|$)",
    re.I | re.S,
)
_SQL_ASSIGN = re.compile(
    r"(?:^|,)\s*\"?(\w+)\"?\s*=\s*('(?:[^']*)'|%s|%\(\w+\)s|:\w+|[\w.]+)", re.S
)


def is_state_field(name: str, has_choices: bool) -> bool:
    return has_choices or name in ("status", "state") or name.endswith("_status")


@dataclass
class ModelInfo:
    name: str
    module: str
    bases: list[str]
    fields: dict[str, list[str]] = field(default_factory=dict)  # field -> choice values
    has_choices: set[str] = field(default_factory=set)
    db_table: Optional[str] = None
    abstract: bool = False

    def state_fields(self) -> set[str]:
        return {f for f in self.fields if is_state_field(f, f in self.has_choices)}


@dataclass
class Write:
    model: Optional[str]
    attribute: str
    outcome: str
    tier: int
    owner: str
    rel_path: str
    line: int
    path_condition: list[str]
    attribution: str
    structural: Optional[str] = None
    form: str = "attribute"


# --------------------------------------------------------------------------- helpers


def _dotted(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _chain(node: ast.AST) -> tuple[Optional[str], list[str]]:
    """``Model.objects.filter(..).update(..)`` -> (``Model``, [filter, update])."""
    methods: list[str] = []
    while True:
        if isinstance(node, ast.Call):
            node = node.func
            continue
        if isinstance(node, ast.Attribute):
            if node.attr == "objects" and isinstance(node.value, ast.Name):
                return node.value.id, list(reversed(methods))
            methods.append(node.attr)
            node = node.value
            continue
        if isinstance(node, ast.Name):
            return None, list(reversed(methods)) + [f"<{node.id}>"]
        return None, []


def _calls_in_chain(node: ast.AST) -> Iterator[ast.Call]:
    while True:
        if isinstance(node, ast.Call):
            yield node
            node = node.func
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            return


def _literal_values(
    node: ast.AST, constants: dict[str, list[str]]
) -> Optional[list[str]]:
    """A choices list or ``__in`` list as plain values, if the code states them."""
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
                elt = elt.elts[0]
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            elif (
                isinstance(elt, ast.Name)
                and elt.id in constants
                and len(constants[elt.id]) == 1
            ):
                out.append(constants[elt.id][0])
            else:
                return None
        return out
    return None


def _module_constants(tree: ast.Module) -> dict[str, list[str]]:
    """``NAME = 'x'`` and ``NAME = [('x', ..), ..]`` at module or class level."""
    constants: dict[str, list[str]] = {}

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name, value = stmt.targets[0].id, stmt.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    constants[name] = [value.value]
                else:
                    values = _literal_values(value, constants)
                    if values is not None:
                        constants[name] = values
            elif isinstance(stmt, ast.ClassDef):
                visit(stmt.body)

    visit(tree.body)
    return constants


# --------------------------------------------------------------------------- plugin


class EpicprodCodeMapPlugin(CodeMapPlugin):
    """Builds the swf-epicprod fragment from a checkout."""

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._stamp = ""
        self.models: dict[str, ModelInfo] = {}
        self.writes: list[Write] = []
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

    # ----------------------------------------------------------------- models

    def _collect_models(self, parsed) -> None:
        raw: dict[str, ModelInfo] = {}
        for _rel, module, tree, _text in parsed:
            constants = _module_constants(tree)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = [_dotted(b) or "" for b in node.bases]
                info = ModelInfo(node.name, module, bases)
                for stmt in node.body:
                    if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
                        for m in stmt.body:
                            if isinstance(m, ast.Assign) and isinstance(
                                m.targets[0], ast.Name
                            ):
                                if m.targets[0].id == "db_table" and isinstance(
                                    m.value, ast.Constant
                                ):
                                    info.db_table = m.value.value
                                if m.targets[0].id == "abstract" and isinstance(
                                    m.value, ast.Constant
                                ):
                                    info.abstract = bool(m.value.value)
                    target = None
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                    ):
                        target, value = stmt.targets[0].id, stmt.value
                    elif (
                        isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.value
                    ):
                        target, value = stmt.target.id, stmt.value
                    if target is None or not isinstance(value, ast.Call):
                        continue
                    func = _dotted(value.func) or ""
                    if not (
                        func.startswith("models.")
                        and (func.endswith("Field") or func.endswith("Key"))
                    ):
                        continue
                    choices: list[str] = []
                    for kw in value.keywords:
                        if kw.arg == "choices":
                            info.has_choices.add(target)
                            choices = _literal_values(kw.value, constants) or []
                            if not choices and isinstance(kw.value, ast.Attribute):
                                choices = constants.get(kw.value.attr, [])
                    info.fields[target] = choices
                raw[node.name] = info

        def is_model(name: str, seen=()) -> bool:
            info = raw.get(name)
            if info is None or name in seen:
                return False
            return any(
                b in ("models.Model", "Model")
                or is_model(b.split(".")[-1], seen + (name,))
                for b in info.bases
            )

        for name, info in raw.items():
            if not is_model(name):
                continue
            # Inherit fields from model ancestors (abstract bases such as PermanentIdentity).
            stack = [b.split(".")[-1] for b in info.bases]
            while stack:
                parent = raw.get(stack.pop())
                if parent is None:
                    continue
                for f, ch in parent.fields.items():
                    info.fields.setdefault(f, ch)
                info.has_choices |= parent.has_choices
                stack.extend(b.split(".")[-1] for b in parent.bases)
            if info.db_table is None:
                info.db_table = f"{info.module.split('.')[0]}_{name.lower()}"
            self.models[name] = info

    # ----------------------------------------------------------------- writes

    def _attribute_classes(self, attrs: set[str]) -> list[str]:
        return sorted(
            m.name
            for m in self.models.values()
            if not m.abstract and attrs <= set(m.fields)
        )

    def _heuristic(self, var: str, attr: str) -> Optional[str]:
        stem = var.lower().lstrip("_").rstrip("s")
        if len(stem) < 3:
            return None
        hits = [
            m.name
            for m in self.models.values()
            if not m.abstract and attr in m.fields and m.name.lower().endswith(stem)
        ]
        return hits[0] if len(hits) == 1 else None

    def _outcome(
        self, value: ast.AST, constants: dict[str, list[str]]
    ) -> tuple[str, int]:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value, 1
        if isinstance(value, ast.Name) and len(constants.get(value.id, [])) == 1:
            return constants[value.id][0], 1
        return f"passthrough({ast.unparse(value)[:80]})", 2

    def _scan_function(
        self,
        func,
        qual: str,
        cls: Optional[str],
        rel: str,
        module: str,
        constants: dict[str, list[str]],
        panda_aliases: set[str],
    ) -> None:
        owner = f"{module}::{qual}"
        bound: dict[str, str] = {}  # var -> model (one object)
        querysets: dict[str, str] = {}  # var -> model (a queryset)
        touched: dict[str, set[str]] = {}  # var -> attributes touched

        if cls in self.models:
            for a in func.args.args[:1]:
                bound[a.arg] = cls
        for a in func.args.args + func.args.kwonlyargs:
            if a.annotation is not None:
                ann = (_dotted(a.annotation) or "").split(".")[-1]
                if ann in self.models:
                    bound[a.arg] = ann

        def bind(target: ast.AST, value: ast.AST) -> None:
            if isinstance(value, ast.Call):
                model, methods = _chain(value)
                fname = (_dotted(value.func) or "").split(".")[-1]
                if model is None and fname in self.models:
                    model, methods = fname, ["<init>"]
                if model is None and fname == "get_object_or_404" and value.args:
                    model = (_dotted(value.args[0]) or "").split(".")[-1] or None
                    methods = ["get"]
                if model in self.models:
                    single = (
                        not methods
                        or methods[-1] in _SINGLE_OBJECT
                        or methods[-1] == "<init>"
                    )
                    if (
                        isinstance(target, ast.Tuple)
                        and methods
                        and methods[-1] in ("get_or_create", "update_or_create")
                    ):
                        if target.elts and isinstance(target.elts[0], ast.Name):
                            bound[target.elts[0].id] = model
                    elif isinstance(target, ast.Name):
                        (bound if single else querysets)[target.id] = model
                        (querysets if single else bound).pop(target.id, None)
                    return
            if isinstance(target, ast.Name):
                bound.pop(target.id, None)
                querysets.pop(target.id, None)

        # Walk with the If-ancestry kept, so each write carries its path condition.
        def walk(stmts: list[ast.stmt], conds: list[str]) -> None:
            for stmt in stmts:
                if isinstance(
                    stmt,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                ):
                    continue
                visit_stmt(stmt, conds)
                if isinstance(stmt, ast.If):
                    test = ast.unparse(stmt.test)
                    walk(stmt.body, conds + [test])
                    walk(stmt.orelse, conds + [f"not ({test})"])
                elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                    model, methods = _chain(stmt.iter)
                    if model in self.models and isinstance(stmt.target, ast.Name):
                        bound[stmt.target.id] = model
                    elif (
                        isinstance(stmt.iter, ast.Name)
                        and stmt.iter.id in querysets
                        and isinstance(stmt.target, ast.Name)
                    ):
                        bound[stmt.target.id] = querysets[stmt.iter.id]
                    walk(stmt.body, conds)
                    walk(stmt.orelse, conds)
                elif isinstance(stmt, (ast.While,)):
                    walk(stmt.body, conds)
                    walk(stmt.orelse, conds)
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    walk(stmt.body, conds)
                elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
                    walk(stmt.body, conds)
                    for h in stmt.handlers:
                        walk(
                            h.body,
                            conds
                            + [
                                f"except {ast.unparse(h.type) if h.type else ''}".strip()
                            ],
                        )
                    walk(stmt.orelse, conds)
                    walk(stmt.finalbody, conds)
                elif isinstance(stmt, ast.Match):
                    for case in stmt.cases:
                        walk(case.body, conds + [f"case {ast.unparse(case.pattern)}"])

        def expressions(stmt: ast.stmt) -> Iterator[ast.AST]:
            """The statement's own expressions, not those of nested blocks."""
            for name, value in ast.iter_fields(stmt):
                if name in ("body", "orelse", "finalbody", "handlers", "cases"):
                    continue
                if isinstance(value, ast.AST):
                    yield value
                elif isinstance(value, list):
                    yield from (v for v in value if isinstance(v, ast.AST))

        def visit_stmt(stmt: ast.stmt, conds: list[str]) -> None:
            for expr in expressions(stmt):
                for node in ast.walk(expr):
                    if isinstance(node, ast.Attribute) and isinstance(
                        node.value, ast.Name
                    ):
                        touched.setdefault(node.value.id, set()).add(node.attr)
                    if isinstance(node, ast.Call):
                        visit_call(node, conds)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        visit_sql(node.value, node.lineno, conds)
                    if isinstance(node, ast.JoinedStr):
                        text = "".join(
                            v.value if isinstance(v, ast.Constant) else "%s"
                            for v in node.values
                        )
                        visit_sql(text, node.lineno, conds)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(
                        target.value, ast.Name
                    ):
                        record_attr(target, stmt.value, stmt.lineno, conds)
                    else:
                        bind(target, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if isinstance(stmt.target, ast.Attribute) and isinstance(
                    stmt.target.value, ast.Name
                ):
                    record_attr(stmt.target, stmt.value, stmt.lineno, conds)
                else:
                    bind(stmt.target, stmt.value)

        pending: list[tuple[str, str, ast.AST, int, list[str], Optional[str]]] = []

        def record_attr(
            target: ast.Attribute, value: ast.AST, line: int, conds: list[str]
        ) -> None:
            var = target.value.id
            # Attribution is settled after the walk, once every touched attribute is known,
            # but the binding is captured now: a later rebinding must not re-attribute it.
            pending.append((var, target.attr, value, line, list(conds), bound.get(var)))

        def visit_call(call: ast.Call, conds: list[str]) -> None:
            fname = call.func.attr if isinstance(call.func, ast.Attribute) else None
            dotted = _dotted(call.func) or ""
            # PanDA boundary: pandaclient Client.* / requests to PANDA_URL.
            head = dotted.split(".")[0]
            if head in panda_aliases or dotted.startswith("pandaclient."):
                method = dotted.split(".")[-1]
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
            elif (
                dotted in ("requests.get", "requests.post", "requests.put")
                and call.args
            ):
                url = call.args[0]
                if isinstance(url, ast.JoinedStr) and any(
                    isinstance(v, ast.FormattedValue)
                    and "PANDA" in ast.unparse(v.value)
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
                        verb=dotted.split(".")[-1].upper(),
                    )
            if fname is None:
                return
            model, methods = _chain(call)
            if model is None and methods and methods[-1].startswith("<"):
                var = methods[-1][1:-1]
                model = querysets.get(var)
                methods = methods[:-1]
            if model not in self.models:
                return
            info = self.models[model]
            # Read side: values a filter selects rows on.
            for c in _calls_in_chain(call):
                if isinstance(c.func, ast.Attribute) and c.func.attr in _ORM_SELECT:
                    for kw in c.keywords:
                        if not kw.arg:
                            continue
                        fld, _, lookup = kw.arg.partition("__")
                        if fld not in info.state_fields():
                            continue
                        if lookup == "in":
                            vals = _literal_values(kw.value, constants) or []
                        elif not lookup:
                            out, tier = self._outcome(kw.value, constants)
                            vals = [out] if tier == 1 else []
                        else:
                            vals = []
                        self.selected.setdefault((model, fld), set()).update(vals)
            if fname not in _ORM_WRITE:
                return
            kwargs: list[tuple[str, ast.AST]] = []
            for kw in call.keywords:
                if kw.arg == "defaults" and isinstance(kw.value, ast.Dict):
                    for k, v in zip(kw.value.keys, kw.value.values, strict=True):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            kwargs.append((k.value, v))
                elif kw.arg and "__" not in kw.arg:
                    kwargs.append((kw.arg, kw.value))
            for fld, value in kwargs:
                if fld not in info.state_fields():
                    continue
                out, tier = self._outcome(value, constants)
                self.writes.append(
                    Write(
                        model,
                        fld,
                        out,
                        tier,
                        owner,
                        rel,
                        call.lineno,
                        list(conds),
                        "certain",
                        form=f"orm.{fname}",
                    )
                )

        def visit_sql(text: str, line: int, conds: list[str]) -> None:
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
                    if raw_value.startswith("'"):
                        out, tier = raw_value.strip("'"), 1
                    else:
                        out, tier = f"passthrough({raw_value})", 2
                    self.writes.append(
                        Write(
                            model,
                            col,
                            out,
                            tier,
                            owner,
                            rel,
                            line,
                            list(conds),
                            "certain",
                            form="sql.update",
                        )
                    )

        walk(func.body, [])

        all_fields = {f for m in self.models.values() for f in m.fields}
        for var, attr, value, line, conds, at_bind in pending:
            if var == "self" and at_bind is None:
                continue  # an attribute of a non-model class: not a row write
            classes = self._attribute_classes(
                (touched.get(var, set()) & all_fields) | {attr}
            )
            structural = classes[0] if len(classes) == 1 else None
            if at_bind is not None:
                model, how = at_bind, "certain"
            elif structural is not None:
                model, how = structural, "structural"
            else:
                model = self._heuristic(var, attr)
                how = "heuristic" if model else "unresolved"
            if model is not None and attr not in self.models[model].state_fields():
                continue
            if model is None and not is_state_field(attr, False):
                continue
            out, tier = self._outcome(value, constants)
            self.writes.append(
                Write(
                    model,
                    attr,
                    out,
                    tier,
                    owner,
                    rel,
                    line,
                    conds,
                    how,
                    structural=structural,
                )
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
        self._collect_models(parsed)

        for rel, module, tree, _text in parsed:
            constants = _module_constants(tree)
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
                self._scan_function(
                    func, qual, cls, rel, module, constants, panda_aliases
                )

        return self._assemble(shas)

    def _assemble(self, shas: dict[str, str]) -> MapFragment:
        subjects: dict[str, SubjectNode] = {}
        entities: dict[str, EntityNode] = {}
        groups: dict[tuple[str, str], list[Write]] = {}
        for w in self.writes:
            subject = f"{w.model}.{w.attribute}" if w.model else f"?.{w.attribute}"
            groups.setdefault((w.owner, subject), []).append(w)
            if w.model is None:
                self.unresolved_writes += 1
                continue
            info = self.models[w.model]
            if subject not in subjects:
                subjects[subject] = SubjectNode(
                    name=subject,
                    map_id=MAP_ID,
                    derived_from=self._stamp,
                    spec_class=w.model,
                    attribute=w.attribute,
                    vocabulary=list(info.fields.get(w.attribute, [])),
                    selected_values=sorted(
                        self.selected.get((w.model, w.attribute), set())
                    ),
                    metadata={"table": info.db_table, "module": info.module},
                )
            entity = entities.setdefault(
                w.model,
                EntityNode(
                    name=f"{MAP_ID}::{w.model}",
                    map_id=MAP_ID,
                    derived_from=self._stamp,
                    spec_class=w.model,
                    tables=[info.db_table] if info.db_table else [],
                ),
            )
            bucket = (
                entity.created_by
                if w.form in ("orm.create", "orm.get_or_create")
                else entity.updated_by
            )
            if w.owner not in bucket:
                bucket.append(w.owner)

        junctions: list[JunctionNode] = []
        for (owner, subject), writes in sorted(groups.items()):
            writes.sort(key=lambda w: w.line)
            first = writes[0]
            # Weakest attribution among the arms; the gate compares against the structural reading.
            rank = ["certain", "structural", "heuristic", "unresolved"]
            attribution = max((w.attribution for w in writes), key=rank.index)
            structural = {w.structural for w in writes if w.structural}
            junctions.append(
                JunctionNode(
                    name=f"{MAP_ID}::{subject}@{owner}",
                    map_id=MAP_ID,
                    derived_from=self._stamp,
                    subject=subject,
                    owner=owner,
                    attribution=attribution,
                    structural_subject=f"{structural.pop()}.{first.attribute}"
                    if len(structural) == 1
                    else None,
                    anchor=Anchor(
                        package=first.rel_path.split("/")[0],
                        file=first.rel_path,
                        line_start=first.line,
                        line_end=writes[-1].line,
                        blob_sha=shas.get(first.rel_path),
                    ),
                    branches=[
                        Branch(
                            outcome=w.outcome,
                            path_condition=w.path_condition,
                            order=i,
                            tier=w.tier,
                            line=w.line,
                            tags=[w.form],
                        )
                        for i, w in enumerate(writes)
                    ],
                )
            )

        return MapFragment(
            map_id=MAP_ID,
            derived_from=self._stamp,
            subjects=sorted(subjects.values(), key=lambda s: s.name),
            entities=sorted(entities.values(), key=lambda e: e.name),
            junctions=junctions,
            boundaries=sorted(self.boundaries.values(), key=lambda b: b.name),
        )
