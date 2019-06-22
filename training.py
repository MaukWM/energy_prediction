import os
import sys

from models.ann import Ann
from models.seq2seq import Seq2Seq
from models.seq2seq_1dconv import Seq2SeqConv
from models.seq2seq_1dconv_attention import Seq2SeqConvAttention
from models.seq2seq_attention import Seq2SeqAttention
from utils import load_data

# Define paths to data of different aggregation sizes
data_path = "/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/"

data_75a_path = os.path.join(data_path, "aggregated_input_data-f83-ak75-b121.pkl")
data_50a_path = os.path.join(data_path, "aggregated_input_data-f83-ak50-b121.pkl")
data_25a_path = os.path.join(data_path, "aggregated_input_data-f83-ak25-b121.pkl")
data_1a_path = os.path.join(data_path, "aggregated_input_data-f83-ak1-b121.pkl")

batch_size = 128
state_size = 128
input_feature_amount = 83
output_feature_amount = 1
seq_len_in = 96
seq_len_out = 96
plot_time_steps_view = 96 * 2
steps_per_epoch = 50
epochs = 50
learning_rate = 0.00045
intermediates = 1
plot_loss = False

load_weights = False
if load_weights:
    load_ann_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/ann-l0.00025-tl0.015-vl0.154-i96-o96-e2250-seq2seq.h5"
    load_s2s_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/s2s-l0.00025-ss36-tl0.027-vl0.042-i96-o96-e2250-seq2seq.h5"
    load_s2s_1dconv_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/s2s1dc-l0.00025-ss36-tl0.026-vl0.058-i96-o96-e2250-seq2seq.h5"
    load_s2s_attention_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/as2s-l0.00025-ss36-tl0.025-vl0.047-i96-o96-e2250-seq2seq.h5"
    load_s2s_1dconv_attention_weights_path = "/home/mauk/Workspace/energy_prediction/models/first_time_training_much data/as2s1dc-l0.00025-ss36-tl0.020-vl0.038-i96-o96-e2250-seq2seq.h5"
else:
    load_ann_weights_path = None
    load_s2s_weights_path = None
    load_s2s_1dconv_weights_path = None
    load_s2s_attention_weights_path = None
    load_s2s_1dconv_attention_weights_path = None

if __name__ == "__main__":
    # Init models
    models = []

    # To train model
    to_train = sys.argv[1]

    # The aggregation level to train on
    agg_level = sys.argv[2]

    # Load data
    if agg_level == "75":
        data_dict = load_data(data_75a_path)
    elif agg_level == "50":
        data_dict = load_data(data_50a_path)
    elif agg_level == "25":
        data_dict = load_data(data_25a_path)
    elif agg_level == "1":
        data_dict = load_data(data_1a_path)
    else:
        raise Exception("Please give a valid aggregation size! The size given was {}.".format(agg_level))

    if to_train == "seq2seq":
        model = Seq2Seq(name="seq2seq",
                        data_dict=data_dict,
                        batch_size=batch_size,
                        state_size=state_size,
                        input_feature_amount=input_feature_amount,
                        output_feature_amount=output_feature_amount,
                        seq_len_in=seq_len_in,
                        seq_len_out=seq_len_out,
                        plot_time_steps_view=plot_time_steps_view,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        intermediates=intermediates,
                        plot_loss=plot_loss,
                        load_weights_path=load_s2s_weights_path,
                        agg_level=agg_level
                        )

    elif to_train == "seq2seq_1dconv":
        model = Seq2SeqConv(name="seq2seq_1dconv",
                            data_dict=data_dict,
                            batch_size=batch_size,
                            state_size=state_size,
                            input_feature_amount=input_feature_amount,
                            output_feature_amount=output_feature_amount,
                            seq_len_in=seq_len_in,
                            seq_len_out=seq_len_out,
                            plot_time_steps_view=plot_time_steps_view,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            intermediates=intermediates,
                            plot_loss=plot_loss,
                            load_weights_path=load_s2s_1dconv_weights_path,
                            agg_level=agg_level
                            )
    elif to_train == "ann":
        model = Ann(name="ann",
                    data_dict=data_dict,
                    batch_size=batch_size,
                    state_size=state_size,
                    input_feature_amount=input_feature_amount,
                    output_feature_amount=output_feature_amount,
                    seq_len_in=seq_len_in,
                    seq_len_out=seq_len_out,
                    plot_time_steps_view=plot_time_steps_view,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    intermediates=intermediates,
                    plot_loss=plot_loss,
                    load_weights_path=load_ann_weights_path,
                    agg_level=agg_level
                    )

    elif to_train == "seq2seq_attention":
        model = Seq2SeqAttention(name="seq2seq_attention",
                                 data_dict=data_dict,
                                 batch_size=batch_size,
                                 state_size=state_size,
                                 input_feature_amount=input_feature_amount,
                                 output_feature_amount=output_feature_amount,
                                 seq_len_in=seq_len_in,
                                 seq_len_out=seq_len_out,
                                 plot_time_steps_view=plot_time_steps_view,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 learning_rate=learning_rate,
                                 intermediates=intermediates,
                                 plot_loss=plot_loss,
                                 load_weights_path=load_s2s_attention_weights_path,
                                 agg_level=agg_level
                                 )

    elif to_train == "seq2seq_1dconv_attention":
        model = Seq2SeqConvAttention(name="seq2seq_1dconv_attention",
                                     data_dict=data_dict,
                                     batch_size=batch_size,
                                     state_size=state_size,
                                     input_feature_amount=input_feature_amount,
                                     output_feature_amount=output_feature_amount,
                                     seq_len_in=seq_len_in,
                                     seq_len_out=seq_len_out,
                                     plot_time_steps_view=plot_time_steps_view,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epochs,
                                     learning_rate=learning_rate,
                                     intermediates=intermediates,
                                     plot_loss=plot_loss,
                                     load_weights_path=load_s2s_1dconv_attention_weights_path,
                                     agg_level=agg_level
                                     )

    else:
        raise Exception("Must give a valid model to train! {} is not a valid model.".format(to_train))

    model.train()
