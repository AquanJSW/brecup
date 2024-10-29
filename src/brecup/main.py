#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def get_version():
    import importlib.metadata

    try:
        # To be used in a package
        version = importlib.metadata.version('brecup')
    except:
        version = '0.0'
    return version


parser = argparse.ArgumentParser(
    description="Simplify the BililiveRecorder → biliup-rs workflow."
)
parser.add_argument(
    'config', type=str, help="Path to the config file.", default="config.brecup.yaml"
)
parser.add_argument('-v', '--version', action='version', version=get_version())


def load_config(path: str) -> dict:
    with open(path) as fs:
        return yaml.safe_load(fs)


def get_resolution(video):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video,
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout.decode().strip()


def assign_property_danmaku(record):
    record['danmaku'] = pathlib.Path(record['video']).with_suffix('.ass')
    ass = record['danmaku']
    xml = ass.with_suffix('.xml')
    resolution = get_resolution(record['video'])
    cmd = [
        'danmaku-factory',
        '--ignore-warnings',
        '-o',
        'ass',
        ass,
        '-i',
        xml,
        '-r',
        resolution,
        '-d',
        '-1',
        '-O',
        '127',
        '--showmsgbox',
        'FALSE',
    ]
    subprocess.run(cmd, check=True)


def assign_property_output(record, output_dir):
    record['output'] = os.path.join(output_dir, os.path.basename(record['video']))


class Processor:
    def __init__(self, config: dict):
        # Get available devices
        if CUDA_VISIBLE_DEVICES := os.environ.get('CUDA_VISIBLE_DEVICES', ''):
            devices = list(map(int, CUDA_VISIBLE_DEVICES.split('=')[-1].split(',')))
        else:
            cmd = f'nvidia-smi -L | wc -l'
            count = (
                subprocess.run(cmd, shell=True, capture_output=True, check=True)
                .stdout.decode()
                .strip()
            )
            devices = list(range(int(count)))
        logger.info(f'Found {len(devices)} devices')

        # Assign BV if possible
        bv = self._get_bv(config)

        self._device_in_using_flags = [False for _ in devices]
        self._devices = devices
        self._config = config
        self._bv = bv

    def _encode(self, record):
        for i, value in enumerate(self._device_in_using_flags):
            if not value:
                device = i
                logger.info(f'Using device {device} for {record["title"]}')
                self._device_in_using_flags[device] = True
                break
        else:
            raise NotImplementedError(
                "This should not happen, you've launch a job too early"
            )
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "auto",
            "-hwaccel_device",
            str(device),
            "-i",
            record['video'],
            '-vf',
            f'ass={record["danmaku"]}',
            '-c:v',
            'h264_nvenc',
            '-c:a',
            'copy',
            '-b',
            '8192K',
            '-y',
        ]
        if 'ss' in record:
            cmd += ['-ss', record['ss']]
        if 'to' in record:
            cmd += ['-to', record['to']]
        cmd += [record['output']]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Releasing device {device} for {record['title']}")
        self._device_in_using_flags[device] = False
        logger.info(f"Encoding output {record['output']}")

        return record

    def run(self):
        # One task per device shall be enough to exhaust the encoder/decoder
        with ThreadPoolExecutor(len(self._devices)) as encoding_executor:
            encoding_futures = [
                encoding_executor.submit(self._encode, record)
                for record in self._config['records']
                if record['enabled']
            ]
            with ThreadPoolExecutor(1) as uploading_executor:
                uploading_futures = []
                for encoding_future in concurrent.futures.as_completed(
                    encoding_futures
                ):
                    record = encoding_future.result()
                    uploading_futures.append(
                        uploading_executor.submit(self._upload, record)
                    )
                logger.info('Waiting for uploading to finish...')
                concurrent.futures.wait(uploading_futures)
            logger.info('Waiting for encoding to finish...')
            concurrent.futures.wait(encoding_futures)
        logger.info('All tasks are done')

    def _upload(self, record):
        c = self._config
        if not self._bv:
            # Upload the first video
            logger.info(
                f"Uploading the video {record['output']} with title '{c['title']}'"
            )
            cmd = [
                "biliup",
                "-u",
                c['cookies'],
                "upload",
                "--tid",
                str(c['tid']),
                "--title",
                c['title'],
                "--tag",
                c['tag'],
                "--cover",
                c['cover'],
                record['output'],
            ]
            output = (
                subprocess.run(cmd, capture_output=True, check=True)
                .stdout.decode()
                .strip()
            )
            result = re.search(r'BV1\w{9}', output)
            assert result, 'BV not found'
            self._bv = result.group(0)
            logger.info(f"BV {self._bv} is assigned to '{c['title']}'")
            logger.warning(
                "You may want to set the first video's title manually if it "
                "belongs to a series"
            )
            return

        # Append a video
        logger.info(f"Appending video '{record['title']}' to {self._bv}")
        cmd = [
            "biliup",
            "-u",
            c['cookies'],
            "append",
            "-v",
            self._bv,
            "--title",
            record['title'],
            record['output'],
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def _get_bv(self, config: dict):
        cmd = [
            "biliup",
            "-u",
            config['cookies'],
            "list",
        ]
        output = (
            subprocess.run(cmd, capture_output=True, check=True).stdout.decode().strip()
        )
        for line in output.split('\n'):
            if config['title'] in line:
                bv = line.split()[0].strip()
                logger.info(f"Found existing BV {bv} for '{config["title"]}'")
                return bv
        logger.info(f"No existing BV found for '{config["title"]}'")
        return ''


def main():
    global dry_run
    args = parser.parse_args()
    config = load_config(args.config)

    os.makedirs(config['output-dir'], exist_ok=True)

    for record in config['records']:
        assign_property_danmaku(record)
        assign_property_output(record, config['output-dir'])
    processor = Processor(config)
    processor.run()


if __name__ == "__main__":
    main()
