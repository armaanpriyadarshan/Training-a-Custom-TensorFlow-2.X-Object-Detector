import tensorflow as tf
import argparse
# Define model and output directory arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the saved model is located in',
                    default='exported-models/my_tflite_model/saved_model')
parser.add_argument('--output', help='Folder that the tflite model will be written to',
                    default='exported-models/my_tflite_model')
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.model)
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('exported-models/my_tflite_model/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#def representative_dataset_gen():
#  for _ in range(num_calibration_steps):
#    # Get sample input data as a numpy array in a method of your choosing.
#    yield [input]
#converter.representative_dataset = representative_dataset_gen
converter.experimental_new_converter = True
#converter.enable_v1_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

output = args.output + '/model.tflite'
with tf.io.gfile.GFile(output, 'wb') as f:
  f.write(tflite_model)
