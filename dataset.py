import random
from tensorflow.python import client
import tensorflow_federated as tff
import collections
import tensorflow as tf
import numpy as np
import pickle

class Dataset:
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=64, test_batch_size=128):
        self.server_prop = server_prop
        self.inner_rounds = inner_rounds
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def get_client_data(self, client_id):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        return self.prefetch_batch(data, 64)

    def get_server_data(self, server_id):
        data = self.train_data.create_tf_dataset_for_client(server_id)
        return self.prefetch_batch(data, self.train_batch_size)
    
    def get_test_data(self, client_id):
        data = self.test_data.create_tf_dataset_for_client(client_id)
        return self.prefetch_batch(data, self.test_batch_size)

    def get_test_dataset(self):
        # n determines the number of clients to test on
        test_clients = np.random.choice(self.list_clients, 10, replace=False)#int(len(self.list_clients) * p), replace=False)
        test_dataset = [(self.get_client_data(client), self.get_test_data(client)) for client in test_clients]
        return test_dataset

    def get_server_proportion(self):
        return self.server_prop

class Shakespeare_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=4, test_batch_size=4):
        super().__init__(server_prop=server_prop, inner_rounds=inner_rounds, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.datasource='shakespeare'
        
        self.train_data, self.test_data = tff.simulation.datasets.shakespeare.load_data()
        self.SEQ_LENGTH = 100
        self.BATCH_SIZE = 4
        self.BUFFER_SIZE = 100
        self.TEST_BATCH_SIZE = 1

        self.vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=self.vocab, values=tf.constant(list(range(len(self.vocab))), dtype=tf.int64)), default_value=0)

        self.ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(self.vocab), mask_token=None)
        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95))
        
        if self.server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
    
    def get_length_data(self, client_id, batch_size=4):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        snippets = list()
        for snippet in data:
            snippets.append(snippet['snippets'].numpy())
        snippets = b' '.join(snippets).decode('utf-8')

        return len(snippets) // 101 > batch_size

    def get_client_data(self, client_id, batch_size=4):
        while self.get_length_data(client_id) == False:
            client_id = np.random.choice(self.list_clients)
        data = self.train_data.create_tf_dataset_for_client(client_id)
        return self.preprocess(data, batch_size=batch_size)

    def get_server_data(self, server_id):
        while self.get_length_data(server_id) == False:
            server_id = np.random.choice(self.list_clients)
        data = self.train_data.create_tf_dataset_for_client(server_id)
        return self.preprocess(data)

    def get_test_data(self, client_id, batch_size=1):
        data = self.test_data.create_tf_dataset_for_client(client_id)
        return self.preprocess(data, batch_size=batch_size)
    
    def get_test_dataset(self, batch_size=1):
        # n determines the number of clients to test on
        test_clients = list()
        for i in range(10):
            cid = np.random.choice(self.list_clients)
            while self.get_length_data(cid, 4) == False:
                cid = np.random.choice(self.list_clients)
            test_clients.append(cid)
        
        test_dataset = [(self.get_client_data(client, batch_size), self.get_test_data(client, batch_size)) for client in test_clients]
        return test_dataset
        
    def to_ids(self, x):
        #s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(x)
        ids = self.table.lookup(chars)
        return ids

    def preprocess(self, data, batch_size=8):
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text
        
        snippets = list()
        for snippet in data:
            snippets.append(snippet['snippets'].numpy())
        # Return client data as one long string    
        snippets = b' '.join(snippets).decode('utf-8')
        
        all_ids = self.ids_from_chars(tf.strings.unicode_split(snippets, 'UTF-8'))
        dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        dataset = dataset.batch(self.SEQ_LENGTH+1, drop_remainder=True)
        dataset = dataset.map(split_input_target)
        dataset = (dataset
            .shuffle(self.BUFFER_SIZE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        return dataset

class CIFAR100_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=20, test_batch_size=20):
        super().__init__()
        self.classes = 100
        self.datasource='cifar100'
        self.server_prop = server_prop

        self.train_data, self.test_data = tff.simulation.datasets.cifar100.load_data()

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95), replace=False)
        if server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
        self.list_test = self.test_data.client_ids

    def get_test_dataset(self):
        # n determines the number of clients to test on
        test_clients = np.random.choice(self.list_clients, 10, replace=False)
        test_dataset = [self.prefetch_test_val(client) for client in test_clients]
        return test_dataset

    def prefetch_batch(self, data, batch_size=20):
        images, labels = list(), list()
        for d in data:
            image = d['image']
            label = d['label']
            images.append(image)
            labels.append(label)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


    def prefetch_test_val(self, client_id, split=0.2):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        images, labels = list(), list()
        val_images, val_labels = list(), list()
        for idx, d in enumerate(data):
            image = d['image']
            label = d['label']
            if random.random() > split:
                val_images.append(image)
                val_labels.append(label)
            else:
                images.append(image)
                labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=128, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(input_preprocess_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size=128, drop_remainder=False)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, val_dataset
        
class EMNIST_Data(Dataset):
    def __init__(self, server_prop=0):
        super().__init__()
        self.classes = 62
        self.datasource='emnist'
        self.server_prop = server_prop

        self.train_data, self.test_data = tff.simulation.datasets.emnist.load_data(only_digits=False)

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95), replace=False)
        if server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)

    
    def prefetch_batch(self, data, batch_size=20):
        images, labels = list(), list()
        for d in data:
            image = d['pixels']
            label = d['label']
            images.append(image)
            labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

class Omniglot_Data:
    def __init__(self, server_proportion=0, use_clients=True, num_clients=10, classes=5, way=5):
        import tensorflow_datasets as tfds
        use_server = server_proportion > 0
        self.datasource='omniglot'
        self.image_size = 28
        self.channel_size = 1   
        self.classes = classes
        self.way = way
        self.data = {}
        self.server_data = {}
        self.client_names = ['{}_{}'.format('client', i+1) for i in range(num_clients)]
        self.clients = {client : {} for client in self.client_names}
        self.client_labels = {client: [] for client in self.client_names}
        self.server_proportion = int(server_proportion)
        self.list_server = [f'server_{i}' for i in range(int(1623 * server_proportion // 100))]

        ds = tfds.load(self.datasource, split='train', as_supervised=True, shuffle_files=False)

        self.list_clients = [f'client_{i}' for i in range(1, num_clients+1)]

        def extraction(image, label):
            # Scale pixel values 
            image = tf.image.convert_image_dtype(image, tf.float32)

            image = tf.image.rgb_to_grayscale(image)

            image = tf.image.resize(image, (self.image_size, self.image_size))
            return image, label

        for image, label in ds.map(extraction):
            image = image.numpy()
            label = str(label.numpy())
            if use_server:
                rand = random.random()
                if rand < server_proportion/100:
                    if label not in self.server_data:
                        self.server_data[label] = []
                    self.server_data[label].append(image)
                    self.server_labels = list(self.server_data.keys())
                else:
                    if label not in self.data:
                        self.data[label] = []
                    self.data[label].append(image)
                    self.labels = list(self.data.keys())
                    if use_clients:
                        client_name = 'client_{}'.format(random.randint(1, num_clients))
                        if label not in self.clients[client_name]:
                            self.clients[client_name][label] = []
                        self.clients[client_name][label].append(image)
                        self.client_labels[client_name] = list(self.clients[client_name].keys())
            else:
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(image)
                self.labels = list(self.data.keys())
                if use_clients:
                    client_name = 'client_{}'.format(random.randint(1, num_clients))
                    if label not in self.clients[client_name]:
                        self.clients[client_name][label] = []
                    self.clients[client_name][label].append(image)
                    self.client_labels[client_name] = list(self.clients[client_name].keys())

    def get_server_proportion(self):
        return self.server_proportion

    def get_client_data(self, client_name, batch_size=32):
        shots = self.way
        num_classes = self.classes

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, self.image_size, self.image_size, self.channel_size))
        label_subset = random.choices(self.client_labels[client_name], k=num_classes)
        for class_idx, _ in enumerate(label_subset):
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            temp_images[
                class_idx * shots : (class_idx + 1) * shots
            ] = random.choices(self.clients[client_name][label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size)
        return dataset

    def get_server_data(self, client_name, batch_size = 32):
        shots = self.way
        num_classes = self.classes

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, self.image_size, self.image_size, self.channel_size))
        label_subset = random.choices(self.server_labels, k=num_classes)
        for class_idx, _ in enumerate(label_subset):
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            temp_images[
                class_idx * shots : (class_idx + 1) * shots
            ] = random.choices(self.server_data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size)
        return dataset

    def get_test_dataset(self):
        test_data = list()
        for _ in range(10):
            finetune, val = self.get_mini_dataset()
            test_data.append((finetune, val))
        return test_data

    def get_mini_dataset(self, batch_size = 32):
        shots = self.way
        num_classes = self.classes
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, self.image_size, self.image_size, self.channel_size))
        test_labels = np.zeros(shape=(num_classes * shots))
        test_images = np.zeros(shape=(num_classes * shots, self.image_size, self.image_size, self.channel_size))

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            images_to_split = random.choices(self.data[label_subset[class_idx]], k=shots)
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            temp_images[
                class_idx * shots : (class_idx + 1) * shots
            ] = images_to_split

            images_to_split2 = random.choices(self.data[label_subset[class_idx]], k=shots)
            test_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            test_images[
                class_idx * shots : (class_idx + 1) * shots
            ] = images_to_split2
            

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images.astype(np.float32), test_labels.astype(np.int32))
        )

        dataset = dataset.shuffle(100).batch(batch_size)#.repeat(repetitions)
        val_dataset = val_dataset.shuffle(100).batch(batch_size)
        return dataset, dataset

class Stackoverflow_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=16, test_batch_size=16):
        super().__init__(server_prop=server_prop, inner_rounds=inner_rounds, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        with open('bow.pkl', 'rb') as f:
            bow = pickle.load(f)
        self.bow = set(bow)
        self.train_data, _, self.test_data = tff.simulation.datasets.stackoverflow.load_data('/home/aiot/.tff')
        self.classes=10000
        self.datasource='stackoverflow'

        self.ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=bow, mask_token=None, num_oov_indices=0)
        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None, num_oov_indices=0)

        client_ids = self.train_data.client_ids
        np.random.shuffle(client_ids)
        partition = int(len(client_ids) * 0.95)
        self.list_clients = client_ids[:partition]
        if server_prop > 0:
            self.list_server= client_ids[partition:]
            #self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
            self.list_server = np.random.choice(self.list_server, server_prop * 100, replace=False)
    

    def get_client_data(self, client_id, test=False):
        if test:
            return self.preprocess_test(client_id)
        return self.preprocess(client_id)
    
    def get_server_data(self, server_id):
        return self.preprocess(server_id)

    def get_test_data(self, client_id, test=False):
        if test:
            return self.preprocess_test(client_id)
        return self.preprocess(client_id)

    def get_test_dataset(self, p=0.1):
        test_clients = np.random.choice(self.list_clients, 10, replace=False)
        test_dataset = [(self.get_client_data(client, test=True), self.get_test_data(client, test=True)) for client in test_clients]
        return test_dataset

    def preprocess(self, client_id, n=20, batch_size=16):
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[-1]
            return input_text, target_text

        all_ids = None
        while all_ids == None:
            train_data = self.train_data.create_tf_dataset_for_client(client_id)
            sentences = list()
            for i in train_data:
                sentence = i['tokens'].numpy().decode("utf-8")
                if len(sentence.split(' ')) >= 20:
                    sentence = [e for e in sentence.split(' ') if e in self.bow]
                    if len(sentence) >= n + 1:
                        #sentences.append(sentence[:n+1])
                        sentences.append(sentence)
            # sentences = list of sentences with more than n words
            sentences = sentences[:][:1000] # Choose the first 10000 sentences
            for sentence in sentences:
                #sent = tf.expand_dims(tf.strings.join(sentence, separator=' '), axis=0)
                sent = tf.strings.ngrams(sentence, n+1)
                ids = self.ids_from_chars(tf.strings.split(sent))
                if all_ids != None:
                    all_ids = tf.concat([all_ids, ids], axis=0)
                else:
                    all_ids = ids
            if all_ids == None:
                client_id = np.random.choice(self.train_data.client_ids)
       
        dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        dataset = dataset.map(split_input_target)
        dataset = (dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

    def preprocess_test(self, client_id, n=20, batch_size=16):
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[-1]
            return input_text, target_text

        all_ids = None
        while all_ids == None:
            train_data = self.train_data.create_tf_dataset_for_client(client_id)
            sentences = list()
            for i in train_data:
                sentence = i['tokens'].numpy().decode("utf-8")
                if len(sentence.split(' ')) >= 20:
                    sentence = [e for e in sentence.split(' ') if e in self.bow]
                    if len(sentence) >= n + 1:
                        sentences.append(sentence[:n+1])
            # sentences = list of sentences with more than n words
            sentences = sentences[:][:1000] # Choose the first 10000 sentences
            for sentence in sentences:
                sent = tf.expand_dims(tf.strings.join(sentence, separator=' '), axis=0)
                ids = self.ids_from_chars(tf.strings.split(sent))
                if all_ids != None:
                    all_ids = tf.concat([all_ids, ids], axis=0)
                else:
                    all_ids = ids
            if all_ids == None:
                client_id = np.random.choice(self.train_data.client_ids)
       
        dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        dataset = dataset.map(split_input_target)
        dataset = (dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

def input_preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def input_preprocess_cifar(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    return image, label