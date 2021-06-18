#!/bin/bash
# python3 test_model.py cfg/dagger_leader_v2.cfg
# python3 test_model.py cfg/dagger_leader1_v2.cfg
# python3 test_model.py cfg/dagger_leader2_v2.cfg
for i in {1..10..1}
do
    python3 test_model.py cfg/dagger_leader_v4.cfg $i
done