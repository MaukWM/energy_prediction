from models.ann import Ann
from models.main import generate_validation_sample
from models.seq2seq import Seq2Seq
from models.seq2seq_1dconv import Seq2SeqConv
from models.seq2seq_attention import Seq2SeqAttention
from utils import load_data

batch_size = 128
state_size = 32
input_feature_amount = 83
output_feature_amount = 1
seq_len_in = 96
seq_len_out = 96
plot_time_steps_view = 96 * 2
steps_per_epoch = 30
epochs = 3
learning_rate = 0.00075
intermediates = 1
plot_loss = True

if __name__ == "__main__":
    data_dict = load_data(
        "/home/mauk/Workspace/energy_prediction/data/prepared/aggregated_1415/aggregated_input_data-f83-ak75-b121.pkl")

    models = []

    seq2seq = Seq2Seq(data_dict=data_dict,
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
                      plot_loss=plot_loss
                      )

    seq2seq_1dconv = Seq2SeqConv(data_dict=data_dict,
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
                                 plot_loss=plot_loss
                                 )

    ann = Ann(data_dict=data_dict,
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
              plot_loss=plot_loss
              )

    seq2seq_attention = Seq2SeqAttention(data_dict=data_dict,
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
                                         plot_loss=plot_loss
                                         )
    #
    # models.append(seq2seq)
    # models.append(seq2seq_1dconv)
    # models.append(ann)
    models.append(seq2seq_attention)

    predict_x_batches, predict_y_batches, predict_y_batches_prev = generate_validation_sample()

    # for model in models:
    #     model.train()

    for model in models:
        model.calculate_accuracy(predict_x_batches, predict_y_batches)
        model.predict(predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)
    # seq2seq.predict(predict_x_batches[0], predict_x_batches[1], predict_y_batches, predict_y_batches_prev)
