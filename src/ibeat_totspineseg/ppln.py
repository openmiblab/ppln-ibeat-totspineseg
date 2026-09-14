import ibeat_mt as ppln
from miblab import pipe

from utils import data

PIPELINE = data.get_pipeline(__file__)

def run(build, logfile):
    
    ppln.stage_01_auto_segment.run(build, logfile)
    ppln.stage_02_manual_segment.run(build, logfile)
    ppln.stage_03_display.run(build, logfile)
    ppln.stage_04_measure.run(build, logfile)


if __name__=='__main__':

    BUILD = data.get_output_buildpath(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()
    
    pipe.run_ppln(run, args.build, PIPELINE)
