import pytest

from kernelrl.tasks import all_tasks, load_task
from kernelrl.tasks.validate import coverage_problems, validate_task

TASKS = all_tasks()


def test_task_counts():
    assert len(TASKS) == 30
    assert [sum(t.level == lv for t in TASKS) for lv in (1, 2, 3)] == [12, 12, 6]


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_schema_and_coverage(task):
    assert task.hidden_shapes()
    assert coverage_problems(task) == []
    assert "KernelBench" in task.provenance and "MIT" in task.provenance
    mod = task.load_module()
    assert callable(mod.reference) and callable(mod.make_inputs)


@pytest.mark.parametrize("tid", ["l1_vector_add", "l2_softmax", "l2_cross_entropy", "l3_attention_causal"])
def test_validator_cpu_smoke(tid):
    rec = validate_task(load_task(tid), "cpu", max_work=3e6)
    assert rec["ok"], rec
