python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 0 --policy_name fleet_manager > results/policies/logs/fleet_manager/test.txt & 
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 1 --policy_name tp_d --alpha 0.0 > results/policies/logs/tp_d/test_00.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 1 --policy_name tp_d --alpha 0.1 > results/policies/logs/tp_d/test_01.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 1 --policy_name tp_d --alpha 0.2 > results/policies/logs/tp_d/test_02.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 2 --policy_name d_tpts --alpha 0.0 > results/policies/logs/d_tpts/test_00.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 2 --policy_name d_tpts --alpha 0.1 > results/policies/logs/d_tpts/test_01.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 2 --policy_name d_tpts --alpha 0.2 > results/policies/logs/d_tpts/test_02.txt &
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 4 --policy_name sequential_greedy > results/policies/logs/sequential_greedy/test.txt & 
python -m HTAMP.evaluate_assignment --use_saved_data --use_saved_request_data --mode 4 --policy_name sequential_greedy --allow_deallocation > results/policies/logs/sequential_greedy/test_ropt.txt & 

wait
echo "Finished running fleet_manager for mode 0"