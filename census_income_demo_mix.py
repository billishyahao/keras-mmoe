"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import roc_auc_score

from mmoe import MMoE

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.compat.v1.random.set_random_seed(SEED)

CAT_VOCAB_SIZE = [
          1461, 586, 11299105, 2416541, 306, 24, 12598, 634, 4, 95980, 5725, 9292738, 3208, 28,
            15211, 6047969, 11, 5722, 2178, 4, 7822987, 18, 16, 303075, 105, 148165
            ]

sparse_vocab_dict = {}
for i in range(1, len(CAT_VOCAB_SIZE)+1):
    sparse_vocab_dict["C" + str(i)] = CAT_VOCAB_SIZE[i-1]
print(sparse_vocab_dict)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def data_preparation():
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        'data/census-income.data.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        'data/census-income.test.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    # One-hot encoding categorical columns
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)

    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def data_preparation_criteo():
    column_names = [ "Id", "Label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12","I13",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
    "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        '/home/yahao/task/mmoe/kaggle-2014-criteo/train.tiny.csv',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        '/home/yahao/task/mmoe/kaggle-2014-criteo/test.tiny.csv',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    # First group of tasks according to the paper
    label_columns = ['Label']

    # One-hot encoding categorical columns
    categorical_columns = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]

    print("yahao: ", train_df.drop(set(column_names)-set(categorical_columns), axis=1).shape)
    transformed_train = train_df.drop(set(column_names)-set(categorical_columns), axis=1)
    transformed_other = other_df.drop(set(column_names)-set(categorical_columns), axis=1)

    # One-hot encoding categorical labels
    '''
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    '''
    train_ctr = to_categorical((train_raw_labels.Label == '0').astype(int), num_classes=2)
    print("train_ctr", train_ctr.shape)
    other_ctr = to_categorical((other_raw_labels.Label == '0').astype(int), num_classes=2)

    dict_outputs = {
        'ctr': train_ctr.shape[1]
    }

    dict_train_labels = {
        'ctr': train_ctr
    }
    dict_other_labels = {
        'ctr': other_ctr
    }

    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def myinput1(inputs, sparse_vocab_dict, sparse_scope, embedding_size, prefix="", is_export=False, name=""):
    with tf.variable_scope(sparse_scope):
        for name in sparse_vocab_dict.keys():
            ## !!! yahao todo 
            # ids = list(inputs[name])
            ids = inputs
            embeddings_name = "embeddings_{prefix}{name}".format(prefix=prefix, name=name)
            if not self.is_export:
                vocab_size = sparse_vocab_dict.get(name)
                #partitioner = tf.fixed_size_partitioner(self.ps_count) if self.ps_count > 0 else None

                embeddings = tf.get_variable(
                        name=embeddings_name,
                        shape=[vocab_size, embedding_size],
                        trainable=True,
                        initializer=tf.constant_initializer(0.1),
                        #partitioner=partitioner,
                        partitioner=None,
                        )
                embeded_get = tf.nn.embedding_lookup(embeddings, ids)
            else:
                embeded_get = tf.nn.embedding_lookup(embeddings_name, tf.cast(tf.reshape(ids, [-1, 1]), tf.int64))
            self.embedded_results.append(embeded_get)
    emb_lookup_results = tf.concat(self.embeded_get, axis=1)
    ## tf.reshape(emb_lookup_results)
    return emb_lookup_results


def myinput(inputs, sparse_vocab_dict, sparse_scope, embedding_size, prefix="", is_export=False, name=""):
    vocab_size = 1000000
    embedded_results = []
    with tf.variable_scope(sparse_scope):
        embeddings_name = "embeddings_{prefix}{name}".format(prefix=prefix, name=name)
        if not is_export:
            #partitioner = tf.fixed_size_partitioner(self.ps_count) if self.ps_count > 0 else None

            embeddings = tf.get_variable(
                    name=embeddings_name,
                    shape=[vocab_size, embedding_size],
                    trainable=True,
                    initializer=tf.constant_initializer(0.1),
                    #partitioner=partitioner,
                    partitioner=None,
                    )
            embeded_get = tf.nn.embedding_lookup(embeddings, inputs)
        else:
            embeded_get = tf.nn.embedding_lookup(embeddings_name, tf.cast(tf.reshape(ids, [-1, 1]), tf.int64))
        embedded_results.append(embeded_get)
    emb_lookup_results = tf.concat(embeded_get, axis=1)
    print("yahao-dbg: emb_lookup_results.shape: ", emb_lookup_results.shape)
    emb_lookup_results = tf.reshape(emb_lookup_results, [-1, emb_lookup_results.shape[1]*emb_lookup_results.shape[2]])
    print("yahao-dbg: after reshaping, emb_lookup_results.shape: ", emb_lookup_results.shape)
    return emb_lookup_results


def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation_criteo()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    # input_layer = Input(shape=(num_features,))

    inputs = tf.placeholder(tf.int32, shape=(None, 26), name="inputs")
    labels = tf.placeholder(tf.float32, shape=(None, 2), name="labels")

    input_layer = myinput(inputs=inputs, sparse_vocab_dict=sparse_vocab_dict, sparse_scope="yahao-",
            embedding_size=16)

    print("yahao-dbg: input_layer.shape: ", input_layer.shape)

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=1
    )(input_layer)

    output_layers = []

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    print("yahao-dbg: output_layers.shape:", output_layers[0].shape)
    print("yahao-dbg: len(output_layers)=", len(output_layers))

    loss = binary_crossentropy(labels, output_layers[0])
    train_op = tf.train.AdamOptimizer().minimize(loss)
    loss_reduce = tf.reduce_mean(loss)
    #train_step = tf.keras.optimizers.Adam().minimize(loss)

    step = 10000
    one_batch = np.random.randint(10000, size=(32, 26))
    # one-hot label tensor
    one_batch_label = np.zeros((32, 2))
    J = np.random.choice(2, 32)
    one_batch_label[np.arange(32), J] = 1


    print("yahao-dbg: one_batch_label:", one_batch_label)

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        for i in range(step):
            _, loss_out = sess.run([train_op, loss_reduce], feed_dict={inputs: one_batch, labels: one_batch_label})
            if i % 100 == 0:
                print("step: {}, loss = {}".format(i, loss_out))


    '''
    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'ctr': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=100
    )
    '''


if __name__ == '__main__':
    main()
