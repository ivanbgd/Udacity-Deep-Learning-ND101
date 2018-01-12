import math

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    
    # TODO: Implement batching
    num_batches = math.ceil(len(features) / batch_size)
    batches = []
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batches.append([features[batch_start: batch_start + batch_size], labels[batch_start: batch_start + batch_size]])

    return batches
