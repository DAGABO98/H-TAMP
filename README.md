# H-TAMP
Repo for Heterogeneous Task and Motion Planning

## Commands

To generate traversal graph:
```console
python -m HTAMP.environment.traversal_graph_gen
```

To generate occupancy reservations:
```console
python -m HTAMP.environment.grid_world
```

To test motion planning:
```console
python -m HTAMP.planning.motion_planner --use_saved_data --num_robots 5
```

To test movement visualization:
```console
python -m HTAMP.planning.state --use_saved_data --num_robots 5
```

To preprocess the data:
```console
python -m HTAMP.data_processing.data_processing
```

To create mapping from medicine names to ATC3 codes:
```console
python -m HTAMP.prediction.medication_mapping.build_medication_mapping --input_csv data/processed/medication_orders_annotated.csv --med_name_col "Medication Generic Name" --output_csv data/prediction/medication_mapping/medication_mapping.csv
```

To debug a specific policy:
```console
python -m HTAMP.assignment.evaluate_assignment --use_saved_data --mode 4 --policy_name sequential_greedy --year 2024 --month 9 --day 30 --floor_number 2  --debug
```

To run all policies:
```console
python -m HTAMP.assignment.run_test
```

To generate U-charts for Distribution Shift Detection:
```console
python -m HTAMP.data_processing.data_statistics
```

To generate team composition results:
```console
python -m HTAMP.team_composition.run_team_comp
```

To plot histograms for team composition results:
```console
python -m HTAMP.plotting.team_comp_plotting
```

To plot box plots for comparison results:
```console
python -m HTAMP.plotting.assignment_results_plotting --daily_stat mean
python -m HTAMP.plotting.assignment_results_plotting --daily_stat p95
```

To train all vital sign models:
```console
python -m HTAMP.prediction.run_vital_sign_tpp_model_comparison   --accelerator gpu   --gpu_ids 0,1,2   --max_parallel_runs 3   --wandb_project final_vital_sign   --run_prefix final_vital_sign   --max_epochs 300 --patience 50   --num_workers 20 --wandb_init_timeout 600
```

To train all medicine delivery models:
```console
python -m HTAMP.prediction.run_delivery_tpp_model_comparison   --accelerator gpu   --gpu_ids 0,1,2   --max_parallel_runs 3   --wandb_project final_med  --run_prefix final_med --max_epochs 300 --patience 50 --num_workers 20 --wandb_init_timeout 600
```

To evaluate vital sign models using OTD with hard event matching:
```console
python -m HTAMP.prediction.run_vital_sign_tpp_otd_evaluation   --comparison_summary_path data/prediction/vital_sign_tpp_comparison/final_vital_sign_summary.csv   --easy_config_path HTAMP/prediction/configs/config_files/prediction/vital_sign_easy_tpp_training.json   --num_samples 20   --max_future_events 8   --max_sequences 50   --sequence_subset_strategy random   --seed 42   --prefix_stride 8 --run_demand_strata  --demand_gpu_ids 0,1,2   --output_dir data/prediction/vital_sign_tpp_otd_by_demand_hard --wandb_project final_vital_sign  --wandb
```

To evaluate vital sign models using OTD with soft event matching:
```console
python -m HTAMP.prediction.run_vital_sign_tpp_otd_evaluation   --comparison_summary_path data/prediction/vital_sign_tpp_comparison/final_vital_sign_summary.csv   --easy_config_path HTAMP/prediction/configs/config_files/prediction/vital_sign_easy_tpp_training.json   --num_samples 20   --max_future_events 8   --max_sequences 50   --sequence_subset_strategy random   --seed 42   --prefix_stride 8 --run_demand_strata  --demand_gpu_ids 0,1,2   --output_dir data/prediction/vital_sign_tpp_otd_by_demand --wandb_project final_vital_sign  --wandb --soft_type_matching --type_weight 0.5
```

To evaluate medicine delivery models using OTD with hard matching:
```console
python -m HTAMP.prediction.run_delivery_tpp_otd_evaluation   --comparison_summary_path data/prediction/delivery_tpp_comparison/final_med_summary.csv   --easy_config_path HTAMP/prediction/configs/config_files/prediction/delivery_easy_tpp_training.json   --num_samples 20   --max_future_events 8   --max_sequences 200   --sequence_subset_strategy random   --seed 42   --prefix_stride 8 --run_demand_strata   --demand_gpu_ids 0,1,2   --output_dir data/prediction/delivery_tpp_otd_by_demand --wandb_project final_med  --wandb 
```

To cache predictions:
```console
python -m HTAMP.prediction.prediction_handlers.offline_request_prediction_cache --tasks vital_sign,delivery --split test --demand_level all --num_samples 20 --max_future_events 8 --selected_vital_runs final_vital_flex_tpp_stp --selected_delivery_runs final_med_flex_tpp_stp
```
