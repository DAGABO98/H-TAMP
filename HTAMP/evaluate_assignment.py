import argparse
import traceback
from datetime import datetime

from HTAMP.environment.traversal_graph_gen import TraversalGraphGenerator

def evaluate_assignment(args):
    print("Generating Traversal Graph...")

    tg_generator = TraversalGraphGenerator(occupancy_map_path=args.occupancy_map_path,
                                           config_path=args.config_path,
                                           meters_per_pixel=args.meters_per_pixel,
                                           factor=args.factor)


def main():
    parser = argparse.ArgumentParser(prog='evaluate_assignment.py',
                                     description='Evaluate assignment algorithms in a hospital floor environment.')
    # date_operational_range parameters
    parser.add_argument("--year", type=int, dest='year', default=2022, help='Select year of interest.')
    parser.add_argument("--month", type=int, dest='month', default=10, help='Select month of interest.')
    parser.add_argument("--day", type=int, dest='day', default=17, help='Select day of interest.')

    # environment parameters
    parser.add_argument("--config_path", type=str, default="maps/hospital_floor/floor_config.yaml", help="Path to the configuration file")
    parser.add_argument("--occupancy_map_path", type=str, default="maps/hospital_floor/occupancy_map.npy", help="Path to the input occupancy map")
    parser.add_argument("--factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--meters_per_pixel", type=float, default=0.036, help="Meters per pixel in the original image")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the grid world")
    parser.add_argument("--occupancy_reservations_file", type=str, default="data/occupancy_reservations.pkl", help="Path to the occupancy reservations file")
    parser.add_argument("--use_saved_data", action='store_true', help="Whether to use saved occupancy reservations data")

    # simulation parameters
    parser.add_argument("--mode", type=int, dest='mode', default=0, help='Select mode of operation.')
    parser.add_argument("--num_robots", type=int, default=1, help="Number of robots used in the team")
    parser.add_argument("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')

    args = parser.parse_args()

    evaluate_assignment(args)
    

if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")