#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pathlib
import subprocess
import tempfile
import time

import yaml

parser = argparse.ArgumentParser(
    description="Simplify BililiveRecorder → biliup workflow."
)
parser.add_argument(
    'config', type=str, help="Path to the config file.", default="config.brecup.py"
)
parser.add_argument('--dry-run', action='store_true', help="Dry run mode.")


def load_config(path: str) -> dict:
    with open(path) as fs:
        return yaml.safe_load(fs)


dry_run = False


def shell_exec(cmd, dummy_output='') -> str:
    want_output = dummy_output is not ''

    print(cmd)

    if dry_run:
        if want_output:
            return dummy_output
        return ''

    if want_output:
        return (
            subprocess.run(cmd, shell=True, capture_output=want_output, check=True)
            .stdout.decode()
            .strip()
        )

    subprocess.run(cmd, shell=True, capture_output=want_output, check=True)
    return ''


def get_resolution(video):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 '{video}'"
    return shell_exec(cmd, dummy_output="1920x1080")


def generate_danmaku_if_necessary(record):
    if 'danmaku' not in record:
        record['danmaku'] = pathlib.Path(record['video']).with_suffix('.xml')
    xml = record['danmaku']
    ass = pathlib.Path(record['danmaku']).with_suffix('.ass')
    resolution = get_resolution(record['video'])
    cmd = f"danmaku-factory --ignore-warnings -o ass '{ass}' -i '{xml}' -r '{resolution}' -d -1 -O 127 --showmsgbox FALSE"
    shell_exec(cmd)
    record['danmaku'] = ass


def assign_video_output_path(records, output_dir):
    cmd = f'mkdir -p {output_dir}'
    shell_exec(cmd)
    for record in records:
        record['output'] = os.path.join(output_dir, os.path.basename(record['video']))


class VideoEditor:
    def __init__(self, config: dict):
        # Get available devices
        if CUDA_VISIBLE_DEVICES := os.environ.get('CUDA_VISIBLE_DEVICES', ''):
            devices = CUDA_VISIBLE_DEVICES.split('=')[-1].split(',')
        else:
            cmd = f'nvidia-smi -L | wc -l'
            devices = list(range(int(shell_exec(cmd, dummy_output=2))))
        print(f'Found {len(devices)} devices')

        # Assign BV if possible
        bv = self._get_bv(
            config,
        )
        if bv:
            print(f'Found existing collection {bv}')

        self._device_in_using_indicators = [False for _ in devices]
        self._devices = devices
        self._config = config
        self._bv = bv

    def _job(self, record):
        for i, v in enumerate(self._device_in_using_indicators):
            if not v:
                device = i
                print(f'picking device {device} for {record["title"]}')
                self._device_in_using_indicators[device] = True
                break
        else:
            raise NotImplementedError('Too early launch of job')
        log = tempfile.mktemp()
        cmd = f"CUDA_VISIBLE_DEVICES={device} ffmpeg -hwaccel auto -i '{record['video']}' -vf 'ass={record['danmaku']}' -c:v h264_nvenc -c:a copy -b 8192K -y -ss '{record['ss']}' -to '{record['to']}' '{record['output']}' > '{log}' 2>&1"
        print(f'log for {record['title']}: {log}')
        shell_exec(cmd)
        if dry_run:
            time.sleep(1)
        print(f"releasing device {device} for {record['title']}")
        print(f"Output {record['output']}")
        self._device_in_using_indicators[device] = False

        return record

    def danmaku_embedding_and_video_clipping_and_upload(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._devices)
        ) as executor:
            futures = [
                executor.submit(self._job, record) for record in self._config['records']
            ]
            with concurrent.futures.ThreadPoolExecutor(1) as upload_executor:
                upload_futures = []
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    upload_futures.append(upload_executor.submit(self._upload, record))
                concurrent.futures.wait(upload_futures)
            concurrent.futures.wait(futures)

    def _upload(self, r):
        c = self._config
        if not self._bv:
            cmd = f"biliup -u '{c['cookies']}' upload --tid '{c['tid']}' --title '{c['title']}' --tag '{c['tag']}' --cover '{c['cover']}' '{r['output']}'"
            shell_exec(cmd)
            time.sleep(10)
            print('Waiting for BV to be available')
            self._bv = self._get_bv(c)
            return
        cmd = f"biliup -u '{c['cookies']}' append -v '{self._bv}' --title '{r['title']}' '{r['output']}'"
        shell_exec(cmd)

    def _get_bv(self, c):
        cmd = f"biliup -u '{c['cookies']}' list | grep '{c['title']}' | cut -f1"
        bv = shell_exec(cmd, dummy_output='BV1234567').strip()
        print(f'Got BV: {bv}')
        return bv


def main():
    global dry_run
    args = parser.parse_args()
    dry_run = args.dry_run
    config = load_config(args.config)
    [generate_danmaku_if_necessary(record) for record in config['records']]
    assign_video_output_path(config['records'], config['output-dir'])
    video_editor = VideoEditor(config)
    video_editor.danmaku_embedding_and_video_clipping_and_upload()


if __name__ == "__main__":
    main()
