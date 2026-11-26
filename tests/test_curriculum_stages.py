from training.curriculum_stages import (
    DEFAULT_STAGES,
    FIXED_MAP_SEED,
    TRAIN_SEED_MAX,
    TRAIN_SEED_MIN,
    StageCurriculum,
)


def test_stage0_is_one_fixed_map_without_enemies():
    cur = StageCurriculum()
    levels = [cur.sample() for _ in range(20)]
    assert {lv["seed"] for lv in levels} == {FIXED_MAP_SEED}
    assert all(lv["tier"] == 1 and lv["knobs"]["enemies_per_room"] == 0.0 for lv in levels)


def test_advances_only_after_mastery_and_walks_the_ladder():
    cur = StageCurriculum(min_episodes=10, window=10, min_steps_per_stage=0, max_steps_per_stage=10**12)
    seen = [cur.spec.name]
    step = 0
    for _ in range(len(DEFAULT_STAGES) * 3):
        for _ in range(10):
            cur.report_episode(True)
        step += 1000
        cur.maybe_advance(step)
        if cur.spec.name != seen[-1]:
            seen.append(cur.spec.name)
    assert seen == [s.name for s in DEFAULT_STAGES]


def test_does_not_advance_when_failing_until_budget_runs_out():
    cur = StageCurriculum(min_episodes=5, window=5, min_steps_per_stage=0, max_steps_per_stage=5000)
    for _ in range(20):
        cur.report_episode(False)
    assert cur.maybe_advance(1000) is None and cur.stage_idx == 0
    assert cur.maybe_advance(6000) is not None and cur.stage_idx == 1


def test_stage_lock_never_advances_and_seeds_stay_out_of_heldout_range():
    cur = StageCurriculum(stage_lock=2, min_episodes=1, window=2, min_steps_per_stage=0)
    for _ in range(5):
        cur.report_episode(True)
    assert cur.maybe_advance(10**9) is None and cur.stage_idx == 2
    for _ in range(50):
        assert TRAIN_SEED_MIN <= cur.sample()["seed"] <= TRAIN_SEED_MAX
