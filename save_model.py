# downloads and untars model to saved_models directory given hub url
import wget
import tarfile
from absl import app, flags
import ssl

FLAGS = flags.FLAGS

flags.DEFINE_string('url', 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2', 'hub url for model')
flags.DEFINE_string('directory', './saved_models/', 'directory to save model to')

def main(argv):
    ssl._create_default_https_context = ssl._create_unverified_context
    model_name = FLAGS.url.split('tensorflow/')[1].replace('/', '_')
    url = FLAGS.url + '?tf-hub-format=compressed'
    wget.download(url, './saved_models')

    tar_file = tarfile.open(FLAGS.directory + model_name + '.tar.gz')
    tar_file.extractall('./saved_models/' + model_name)

if __name__ == '__main__':
    app.run(main)