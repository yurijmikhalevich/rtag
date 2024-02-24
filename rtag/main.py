import os
from rtag import utils
from rclip.main import init_rclip
from iptcinfo3 import IPTCInfo
import numpy as np
import textwrap
import sys
from tqdm import tqdm


def get_imagenet_tags_filepath():
  return os.path.join(os.path.dirname(__file__), 'data', 'imagenet-labels.txt')


def load_tags_from_file(path: str):
  with open(path, 'r') as f:
    return f.read().splitlines()


def main():
  arg_parser = utils.init_arg_parser()
  args = arg_parser.parse_args()

  if not args.dry_run and not args.yes:
    print(
      textwrap.fill(
        'NOTICE: rtag is under active development. Expect bugs and changes.'
        ' It is recommended for you to have a backup of your images before running rtag.',
        width=utils.get_terminal_text_width(),
      ) +
      '\n\n' +
      textwrap.fill(
        'rtag is about to write new tags to the metadata of the images in the curent directory.'
        ' This operation:',
        width=utils.get_terminal_text_width(),
      ) +
      '\n'
      '  - is irreversible\n'
      '  - will modify the image files\n\n'
      'Continue? [y/n]',
    )
    if input().lower() != 'y':
      sys.exit(10)

  current_directory = os.getcwd()

  _, model, rclipDB = init_rclip(
    current_directory,
    args.indexing_batch_size,
    vars(args).get("device", "cpu"),
    args.exclude_dir,
    args.no_indexing,
  )

  # load tags
  tags_filepath = args.tags_filepath or get_imagenet_tags_filepath()
  tags = load_tags_from_file(tags_filepath)

  # generate features for the tags
  tag_features = model.compute_text_features(tags)

  # loop over the images
  print("tagging images")
  for image in tqdm(rclipDB.get_image_vectors_by_dir_path(current_directory), unit='images'):
    image_path = image['filepath']
    image_vector = np.frombuffer(image['vector'], np.float32)

    similarities = image_vector @ tag_features.T

    if args.dry_run:
      print(f'\n{image_path}')

    new_tags = []
    for tag, similarity in zip(tags, similarities):
      if similarity > args.threshold:
        if args.dry_run:
          print(f'- {tag}: {similarity:.3f}')
        new_tags.append(tag)

    if not new_tags or args.dry_run:
      continue

    image_metadata = IPTCInfo(image_path, force=True)

    if args.mode == 'append':
      existing_tags = image_metadata['keywords']
      image_metadata['keywords'] = [*set(existing_tags + new_tags)]
    elif args.mode == 'overwrite':
      image_metadata['keywords'] = new_tags
    else:
      raise ValueError(f'Invalid mode: {args.mode}')

    image_metadata.save()


if __name__ == '__main__':
  main()
