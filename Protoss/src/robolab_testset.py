"""Shared RoboLab testset for post-training and standalone evaluation."""

TEST_TASKS = (
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "BananasInBinOneMoreTask",
    "BananasInBinThreeTotalTask",
    "BananasInCrateTask",
)

# RoboLab runs num_envs episodes per sequential run: 5 tasks * 2 * 10 = 100.
TEST_NUM_ENVS = 2
TEST_NUM_RUNS = 10
