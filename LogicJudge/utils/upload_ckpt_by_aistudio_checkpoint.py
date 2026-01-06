#-*- coding:utf-8
import os
# !!! note 用别人容器时需设置自己的工号
os.environ["USERNUMBER"] = "466735"
import argparse
from aistudio_checkpoint.aistudio_base_checkpointer import AistudioCheckpointer
from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
from aistudio_checkpoint.logger_utils import default_logger as logger

def upload_ckpt(args):
    ckpter = AistudioCheckpointer(checkpoint_dir=args.ckpt_dir)
    aistudio_path = ckpter.save_checkpoint(path=args.ckpt_dir, step=args.step, memo=args.memo)
    logger.info(f"aistudio_path {aistudio_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_dir', type=str, default='step1000', help='上传的ckpt')
    parser.add_argument('--step', type=int, default=1, help='step数')
    parser.add_argument('--memo', type=str, default=None, help='类似tag，方便用户检索')
    args = parser.parse_args()

    upload_ckpt(args)
