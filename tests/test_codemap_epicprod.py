"""swf-epicprod Code Map plugin: Django-model subjects and ORM/SQL write recognizers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import gates
from bamboo.codemap.epicprod.plugin import EpicprodCodeMapPlugin
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin

MODELS = """
from django.db import models

TASK_STATUS = [('draft', 'Draft'), ('submitted', 'Submitted'), ('done', 'Done')]


class Base(models.Model):
    class Meta:
        abstract = True


class ProdTask(Base):
    status = models.CharField(max_length=20, choices=TASK_STATUS, default='draft')
    name = models.CharField(max_length=20)

    class Meta:
        db_table = 'pcs_prod_task'

    def finish(self):
        self.status = 'done'


class Note(models.Model):
    text = models.TextField()
"""

SERVICES = """
import pandaclient.Client as C
import requests
from .models import ProdTask

PANDA_URL = 'https://panda'
SUBMITTED = 'submitted'


class Helper:
    def __init__(self, status):
        self.status = status


def submit(pk):
    task = ProdTask.objects.select_for_update().get(pk=pk)
    if task.status == 'draft':
        task.status = SUBMITTED
    else:
        task.status = 'draft'
    task.save()


def make():
    ProdTask.objects.create(name='x', status='draft')
    ProdTask.objects.filter(status__in=['submitted']).update(status='done')


def structural(t):
    t.name = 'y'
    t.status = 'submitted'


def raw(cursor, value):
    cursor.execute("UPDATE pcs_prod_task SET status = 'done' WHERE id = %s", [value])


def ask(token):
    C.getJobStatus([1])
    C._Curl()
    requests.get(f'{PANDA_URL}/api/v1/creds/get_access_token', timeout=3)
"""


@pytest.fixture()
def fragment_and_plugin(tmp_path: Path):
    pkg = tmp_path / "pcs"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text(textwrap.dedent(MODELS))
    (pkg / "services.py").write_text(textwrap.dedent(SERVICES))
    (pkg / "migrations").mkdir()
    (pkg / "migrations" / "0001.py").write_text("def f(t):\n    t.status = 'ignored'\n")
    plugin = EpicprodCodeMapPlugin()
    stamp = plugin.prepare(tmp_path)
    assert stamp.startswith("swf-epicprod ")
    return plugin.run(), plugin


def _arms(fragment, owner_suffix):
    return [
        (b.outcome, b.tier, j.attribution)
        for j in fragment.junctions
        if j.owner.endswith(owner_suffix)
        for b in j.branches
    ]


def test_registered():
    assert "swf-epicprod" in available_map_ids()
    assert isinstance(get_code_map_plugin("swf-epicprod"), EpicprodCodeMapPlugin)


def test_prepare_needs_a_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        EpicprodCodeMapPlugin().prepare(None)
    with pytest.raises(FileNotFoundError):
        EpicprodCodeMapPlugin().prepare(tmp_path)


def test_subject_carries_declared_vocabulary(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    subject = next(s for s in fragment.subjects if s.name == "ProdTask.status")
    assert subject.vocabulary == ["draft", "submitted", "done"]
    assert subject.selected_values == ["submitted"]
    assert subject.metadata["table"] == "pcs_prod_task"
    assert {s.name for s in fragment.subjects} == {"ProdTask.status"}


def test_bound_object_writes_are_certain_with_path_conditions(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    junction = next(
        j for j in fragment.junctions if j.owner == "pcs/services.py::submit"
    )
    assert [(b.outcome, b.tier) for b in junction.branches] == [
        ("submitted", 1),
        ("draft", 1),
    ]
    assert junction.attribution == "certain"
    assert junction.branches[0].path_condition == ["task.status == 'draft'"]
    assert junction.branches[1].path_condition == ["not (task.status == 'draft')"]


def test_self_inside_model_is_certain(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    assert _arms(fragment, "ProdTask.finish") == [("done", 1, "certain")]


def test_orm_and_sql_writes(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    assert sorted(_arms(fragment, "::make")) == [
        ("done", 1, "certain"),
        ("draft", 1, "certain"),
    ]
    assert _arms(fragment, "::raw") == [("done", 1, "certain")]


def test_structural_attribution(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    junction = next(j for j in fragment.junctions if j.owner.endswith("::structural"))
    assert junction.attribution == "structural"
    assert junction.structural_subject == "ProdTask.status"


def test_skips_non_model_self_and_migrations(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    owners = {j.owner for j in fragment.junctions}
    assert not any("Helper" in o or "migrations" in o for o in owners)


def test_panda_boundaries(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    assert {b.interface for b in fragment.boundaries} == {
        "pandaclient.Client.getJobStatus",
        "/api/v1/creds/get_access_token",
    }
    assert all(
        b.direction == "outbound" and b.system == "panda" for b in fragment.boundaries
    )


def test_generic_gates_pass(fragment_and_plugin):
    fragment, _ = fragment_and_plugin
    results = {r.gate: r for r in gates.run_all(fragment)}
    for name in (
        "map-references-resolve",
        "branch-positions-are-known",
        "structural-attribution-agrees",
        "declared-status-is-written",
    ):
        assert results[name].passed, results[name].failures
