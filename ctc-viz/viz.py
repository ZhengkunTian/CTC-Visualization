import numpy as np
import matplotlib.pyplot as plt


def draw(probs, index=None, vocab=None, blank=0, save_or_show='show', saved_name=None):
    """
    args:
        probs: the probs from softmax layer of CTC model. dim: [time_steps, vocab_size] or [time_steps, target_length].
        index(optional): list, target index of vocabulary. such as [2, 4, 8, 3]
        vocab(optional): ordered list or dict, the vocabulary in your model. such ['a', 'b'] or {'a': 0, 'b': 1}
        blank: the index of blank in your vocabulary, default: 0.
        save_or_show: 'show', 'save' or 'all'
        saved_name: you can set saved name to store the generated pic
    """
    assert len(
        probs.shape) == 2, 'Please set the dimension of probs as [time_steps, vocab_size] or [time_steps, target_length].'
    assert save_or_show in [
        'show', 'save', 'all'], "Please set value of save_or_show in ['show', 'save', 'all'] "

    if index is not None:
        if blank not in index:
            index.append(blank)

        index.sort()

        if vocab is not None:
            if type(vocab).__name__ == 'list':
                vocab2int = {vocab[i]: i for i in range(len(vocab))}
            else:
                vocab2int = vocab

            int2vocab = {i: v for (v, i) in vocab2int.items()}

        if probs.shape[1] > len(index):
            probs = probs[:, index]

    if index is not None:
        if vocab is not None:
            label = [int2vocab[i] for i in index]
        else:
            label = [str(i) for i in index]
    else:
        label = None

    plt.title('Result Analysis')

    x = list(range(probs.shape[0]))
    for i in range(probs.shape[1]):
        if label is not None:
            if index[i] == blank:
                plt.plot(x, probs[:, i], linestyle='--', label='blank')
            else:
                plt.plot(x, probs[:, i], label=label[i])
        else:
            plt.plot(x, probs[:, i])

    plt.legend()
    plt.xlabel('Frams')
    plt.ylabel('Probs')

    if save_or_show in ['save', 'all']:
        if saved_name is not None:
            plt.savefig(saved_name)
        else:
            plt.savefig('ctc-viz.jpg')

    if save_or_show in ['show', 'all']:
        plt.show()


if __name__ == '__main__':
    probs = np.array([[0.91, 0.03, 0.03, 0.03], [0.91, 0.03, 0.03, 0.03],
                    [0.91, 0.03, 0.03, 0.03], [0.03, 0.91, 0.03, 0.03],
                    [0.91, 0.03, 0.03, 0.03], [0.03, 0.03, 0.91, 0.03],
                    [0.91, 0.03, 0.03, 0.03], [0.03, 0.03, 0.03, 0.91],
                    [0.91, 0.03, 0.03, 0.03], [0.91, 0.03, 0.03, 0.03]]
    )
    index = [1, 2, 3]
    vocab = ['blank', 'a', 'b', 'c']
    draw(probs, index=index, vocab=vocab, blank=0)
