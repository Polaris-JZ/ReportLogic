import argparse
from aistudio_checkpoint.checkpoint_supporter import sync_checkpoint
from aistudio_checkpoint.aistudio_base_checkpointer import download_checkpoint_to_local
from aistudio_checkpoint.logger_utils import default_logger as logger

def download_ckpt(args):
    if args.sync is not None:
        sync_checkpoint(args.resuming_path, args.sync)
    download_path = download_checkpoint_to_local(resuming_path=args.resuming_path, local_path=args.local_path, region=args.region)
    logger.info("download_path value %s" % download_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--resuming_path', type=str, default=None)
    parser.add_argument('--local_path', type=str, default=None)
    parser.add_argument('--region', type=str, default=None)
    parser.add_argument('--sync', type=str, default=None)
    args = parser.parse_args()
    
    download_ckpt(args)