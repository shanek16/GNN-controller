#!/bin/bash
# python3 test_model.py cfg/dagger_leader_v2.cfg
# python3 test_model.py cfg/dagger_leader1_v2.cfg
# python3 test_model.py cfg/dagger_leader2_v2.cfg

# v4
# for i in {1..10..1}
# do
#     python3 test_model.py cfg/dagger_leader_v4.cfg $i
# done

#v2
python3 test_model.py cfg/dagger_leader1_v2.cfg 9.4
for i in {10..100..10}
do
    python3 test_model.py cfg/dagger_leader1_v2.cfg $i
done