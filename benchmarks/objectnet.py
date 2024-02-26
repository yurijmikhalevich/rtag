# This is an image tagging benchmark run against the ObjectNet dataset https://objectnet.dev/
# The ObjectNet dataset is used because it was collected to intentionally show objects from
# new viewpoints on new backgrounds. And the results obtained on the ObjectNet dataset are
# more representative of the performance you can expect in the real world.
#
# You may need to increase the ulimit to avoid "Too many open files" error:
# `ulimit -n 1024`
#
# You may also presize the images to speed up the benchmark via (requires ImageMagick):
# `find . -name "*.png" -exec mogrify -resize 224x224\^ {} \;`

import os
from rclip.main import init_rclip
from tqdm import tqdm
import numpy as np
import tempfile
import json


DATASET_DIR = os.getenv('OBJECTNET_DIR', os.path.join(os.path.dirname(__file__), 'datasets', 'objectnet-1.0'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 256))
DEVICE = os.getenv('DEVICE', 'cpu')


def main(tmp_datadir: str):
  if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f'ObjectNet dataset not found at {DATASET_DIR}')
  
  print(f'Using dataset: {DATASET_DIR}')
  print(f'Batch size: {BATCH_SIZE}')
  print(f'Device: {DEVICE}')

  os.environ['RCLIP_DATADIR'] = tmp_datadir
  _, model, rclipDB = init_rclip(DATASET_DIR, BATCH_SIZE, DEVICE)

  # load objectnet directory-to-tag map from folder_to_objectnet_label.json
  with open(os.path.join(DATASET_DIR, 'mappings', 'folder_to_objectnet_label.json')) as f:
    directory_to_tag = json.load(f)

  def get_tag_from_image_path(image_path: str) -> str:
    return directory_to_tag[os.path.basename(os.path.dirname(image_path))]

  tags = list(directory_to_tag.values())

  # write the tags to file, every one on a new line
  # with open(os.path.join(os.path.dirname(__file__), 'objectnet-tags.txt'), 'w') as f:
  #   f.write('\n'.join(tags))

  # generate features for the tags
  tag_features = model.compute_text_features(tags)
  # tag_features = model.compute_text_features([f'photo of {tag}' for tag in tags])

  top1_match = 0
  top5_match = 0
  processed = 0
  batch = []

  def process_batch():
    nonlocal processed, top1_match, top5_match, batch

    image_features = np.stack([np.frombuffer(image['vector'], np.float32) for image in batch])

    similarities = image_features @ tag_features.T
    ordered_similarities = np.argsort(similarities, axis=1)

    target_classes = np.array([get_tag_from_image_path(image['filepath']) for image in batch])
    top1_match += np.sum(target_classes == tags[ordered_similarities[:, -1]])
    top5_match += np.sum(np.any(target_classes.reshape(-1, 1) == tags[ordered_similarities[:, -5:]], axis=1))

    processed += len(batch)

    batch = []

  for image in tqdm(rclipDB.get_image_vectors_by_dir_path(DATASET_DIR), unit='images'):
    batch.append(image)
    if len(batch) < BATCH_SIZE:
        continue
    process_batch()

  if len(batch) > 0:
    process_batch()

  print(f'Processed: {processed}')
  print(f'Top-1 accuracy: {top1_match / processed}')
  print(f'Top-5 accuracy: {top5_match / processed}')


if __name__ == '__main__':
  with tempfile.TemporaryDirectory() as tmp_dir:
    print(f'Using temporary directory: {tmp_dir}')
    main(tmp_dir)
