"""Shared RoboLab testset for post-training and standalone evaluation."""

TEST_TASKS = (
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "BananasInBinOneMoreTask",
    "BananasInBinThreeTotalTask",
    "BananasInCrateTask",
)

# RoboLab runs num_envs episodes per sequential run: 5 tasks * 1 * 10 = 50.
TEST_NUM_ENVS = 1
TEST_NUM_RUNS = 10
