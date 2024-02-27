# rtag - AI-Powered Command-Line Photo Tagging Tool

### NOTICE: rtag is under active development. Expect bugs and changes.

**rtag** is a command-line photo tagging tool based on [rclip](https://github.com/yurijmikhalevich/rclip) and powered by the awesome OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

## Usage

When run without arguments, **rtag** tags all of the images in the current directory, including
subdirectories, with the most relevant tags out of [the imagenet-1k tag list](https://github.com/yurijmikhalevich/rtag/blob/main/rtag/data/imagenet-labels.txt). **rtag** writes the tags to the
images' metadata, in a format supported by most photo management software. **rtag** supports JPG
images only.

```bash
cd photos && rtag
```

### How do I use my own tag list?

You can use your own tag list by providing a path to a file with tags as an argument to **rtag**.

```bash
cd photos && rtag --tags-filepath /path/to/your/tag/list.txt
```

The tag list file should contain one tag per line.

You can also adjust a `--threshold` used to filter out tags with a lower
confidence score. Run **rtag** in the `--dry-run` mode to see the confidence
scores for each tag that passes the threshold.

## Help

```bash
rtag --help
```

## Contributing

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

### Running locally from the source code

To run **rtag** locally from the source code, you must have [Python](https://www.python.org/downloads/) and [Poetry](https://python-poetry.org/) installed.

Then do:
```bash
# clone the source code repository
git clone git@github.com:yurijmikhalevich/rtag.git

# install dependencies and rtag
cd rtag
poetry install

# activate the new poetry environment
poetry shell
```

If the poetry environment is active, you can use **rtag** locally, as described in the [Usage](#usage) section above.
