import argparse
import os
from importlib.metadata import version
from rclip.const import IS_MACOS
from rclip.utils.helpers import is_mps_available


def positive_int_arg_type(arg: str) -> int:
  arg_int = int(arg)
  if arg_int < 1:
    raise argparse.ArgumentTypeError('should be >0')
  return arg_int


def get_terminal_text_width() -> int:
  try:
    return min(100, os.get_terminal_size().columns - 2)
  except OSError:
    return 100
  

class HelpFormatter(argparse.RawDescriptionHelpFormatter):
  def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 24) -> None:
    text_width = get_terminal_text_width()
    super().__init__(prog, indent_increment, max_help_position, width=text_width)


def init_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    formatter_class=HelpFormatter,
    prefix_chars='-+',
    description='rtag is an AI-powered command-line photo tagging tool',
    epilog='if you like rtag, checkout its sister project rclip - an AI-powered command-line photo search tool:\n' +
    '  https://github.com/yurijmikhalevich/rclip\n\n'
    'get help:\n'
    '  https://github.com/yurijmikhalevich/rtag/discussions/new/choose\n\n',
  )

  version_str = f'rtag {version("rtag")}'
  parser.add_argument('--version', '-v', action='version', version=version_str, help=f'prints "{version_str}"')

  parser.add_argument(
    '--dry-run',
    action='store_true',
    default=False,
    help='do not write any changes into the images; print them to the console instead',
  )
  parser.add_argument('--tags-filepath', help='path to the file with the newline-separated tags; default: imagenet-1k tags')
  parser.add_argument('--yes', '-y', action='store_true', default=False, help='do not ask for confirmation')
  parser.add_argument('--mode', '-m', choices=['append', 'overwrite'], default='append', help='default: append')
  parser.add_argument(
    '--threshold',
    '-t',
    type=float,
    default=0.25,
    help='tag confidence threshold; all tags with the confidence lower than this value will be ignored; default: 0.25',
  )
  parser.add_argument(
    '--no-indexing', '--skip-index', '--skip-indexing', '-n',
    action='store_true',
    default=False,
    help='allows to skip updating the index if no images were added, changed, or removed'
  )
  parser.add_argument(
    '--indexing-batch-size', '-b', type=positive_int_arg_type, default=8,
    help='the size of the image batch used when updating the search index;'
    ' larger values may improve the indexing speed a bit on some hardware but will increase RAM usage; default: 8',
  )
  parser.add_argument(
    '--exclude-dir',
    action='append',
    help='dir to exclude from search, can be used multiple times;'
    ' adding this argument overrides the default of ("@eaDir", "node_modules", ".git");'
    ' WARNING: the default will be removed in v2'
  )
  if IS_MACOS:
    if is_mps_available():
      parser.add_argument('--device', '-d', default='mps', choices=['cpu', 'mps'],
                          help='device to run on; default: mps')

  return parser
