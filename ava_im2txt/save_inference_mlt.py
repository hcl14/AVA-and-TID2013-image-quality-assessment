# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import multiprocessing
import math
import os
import re
import pickle
        
checkpoint_path = '../im2txt_5M'
vocab_file = '../word_counts.txt'
input_files = "../COCO_val2014_000000224477.jpg"


data_path = '../../../datasets/AVA_dataset/images/images'
    

filenames = [f for f in os.listdir(data_path) if re.match(r'[0-9]+.*\.jpg', f)]
    
print('Files found: {}'.format(len(filenames)))

n_processes = os.cpu_count() // 3

array_part_length = len(filenames) // n_processes


def worker(pr_id):
    
    filenames1 = filenames[(pr_id*array_part_length) : ((pr_id+1)*array_part_length)]

    import tensorflow as tf
    '''
    from im2txt import configuration
    from im2txt import inference_wrapper
    from im2txt.inference_utils import caption_generator
    from im2txt.inference_utils import vocabulary
    '''
    import configuration
    import inference_wrapper
    from inference_utils import caption_generator
    from inference_utils import vocabulary


    #FLAGS = tf.flags.FLAGS

    '''
    tf.flags.DEFINE_string("checkpoint_path", "",
                        "Model checkpoint file or directory containing a "
                        "model checkpoint file.")
    tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
    tf.flags.DEFINE_string("input_files", "",
                        "File pattern or comma-separated list of file patterns "
                        "of image files.")
    '''

    tf.logging.set_verbosity(tf.logging.INFO)


    #def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        #restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
        #                                           FLAGS.checkpoint_path)
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                checkpoint_path)
    
    g.finalize()

    # Create the vocabulary.
    #vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    vocab = vocabulary.Vocabulary(vocab_file)


    #for file_pattern in FLAGS.input_files.split(","):
    #for file_pattern in input_files.split(","):
    #  filenames.extend(tf.gfile.Glob(file_pattern))
    
    #tf.logging.info("Running caption generation on %d files matching %s",
    #                len(filenames), FLAGS.input_files)

    processed_data = []

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        j = 0
        for filename in filenames1:
            j += 1
            try:
                with tf.gfile.GFile(os.path.join(data_path,filename), "rb") as f:
                    image = f.read()
                    if j%100==0:
                        print('{}: {}, {}/{}, {}'.format(pr_id, j,len(processed_data), array_part_length, filename))
                    if j%1000==0:
                        print(processed_data[-1])
                    captions = generator.beam_search(sess, image)
                    #print("Captions for image %s:" % os.path.basename(filename))
            except Exception as e:
                print(e)
                continue
            
            
            results = []
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                #sentence = " ".join(sentence)
                results.append([sentence, math.exp(caption.logprob)])
                #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            processed_data.append([int(filename.split(".")[0]), results])
        # save predictions  
        with open('predictions_{}.pickle'.format(pr_id), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    


  

processes=[multiprocessing.Process(target=worker, args=(x,)) for x in range(n_processes)]
  
for p in processes:
    p.start()
    
for p in processes:
    p.join()
  

  #main()
