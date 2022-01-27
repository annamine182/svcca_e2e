##svcca analysis framework
##version: 0.1
## author: Asif, Anna 2020
import numpy as np
import torch
import pickle
import os
import cca_core

def parse_model(model_files, keys, subkey, dst_dir):
    """
    :param model_files: The saved checkpoints.
    :param keys: Main keys for layers. Example: convolutions, lstm, encoder etc
    :param subkey: Specific layer. Key and corresponding subkey should be next to each other.
    Example: 0 for "convolutions.0" or "lstm.0" ..depending on the corresponding key.
    "*" for all possible option with the corresponding key.
    :param dst_dir: The destination files main directory
    :return: layer dictionary where each key has a list of all epochs of the particular layers with that key
    Example: "encoder.conv_layers_before.convolutions.0.weight" key contains all the layer parameters of "encoder.conv_layers_before.convolutions.0.weight".
    Saved in a list in a dictionary format.
    """
    layer_dict = {}
    query = []
    models = []
    for i in range(len(model_files)):
        models.append(torch.load(model_files[i])["model"])
    for i in range(len(keys)):
        if subkey[i] != "*":
            query.append("%s.%s"%(keys[i],subkey[i]))
        else:
            query.append("%s" % (keys[i]))
    # print(query)
    for m in range(len(models)):
        for i in range(len(query)):
            for model_key in models[m]:
                if query[i] in model_key:
                    # print(model_key)
                    if model_key not in layer_dict:
                        layer_dict[model_key] = []
                    layer_dict[model_key].append(models[m][model_key])
    # layer_dict_file = os.path.join(dst_dir,"layer_dict.pickle")
    # torch.save(layer_dict, layer_dict_file)
    return layer_dict
def svcca_analysis(layer_epoch_dictionary, dst_dir=" "):
    # print("Tobe finished later")
    for index in layer_epoch_dictionary:
        layer = layer_epoch_dictionary[index]
        for i in range(len(layer)-1):
            # print("Layer{} --> epoch {} shape {} --- layer{} --> epoch {} shape{} ".format(index,0, layer[0].shape, index, i + 1, layer[i+1].shape))
            if "conv" in index and "bias" not in index:
                height, width, channel1, channel2 = layer[0].shape
                layerA = layer[0].reshape((height*width, channel1*channel2))
                layerB = layer[i+1].reshape((height*width, channel1*channel2))
                #print(layerA.shape, layerB.shape)
                similarity = cca_core.get_cca_similarity(layerA.T.detach().numpy(), layerB.T.detach().numpy(), verbose=True)
                print("Layer {} epoch {} : Similarity {}".format(index, i + 1, np.mean(similarity["cca_coef1"])))
                #f=plt.figure()
                #plt.plot(results["cca_coef1"], lw=2.0)
                #plt.xlabel("CCA coef idx")
                #plt.ylabel("CCA coef value")
                #plt.grid()
                #f.savefig("graphs/test.pdf",bbox_inches='tight')
            elif "lstm" in index and "bias" not in index:
                similarity = cca_core.get_cca_similarity(layer[0].T.detach().numpy(), layer[i + 1].T.detach().numpy(), verbose=False)
                print("Layer {} epoch {} : Similarity {}".format(index, i + 1, np.mean(similarity["cca_coef1"])))
def main_fnc(src_dir, dst_dir, keys, subkeys):
    model_files = []
    files = os.listdir(src_dir)
    for f in files:
        if ".pt" in f:
            model_files.append(os.path.join(src_dir, f))
    model_files.sort()
    # print(model_files)
    epochs_layer_dictionary = parse_model(model_files, keys, subkeys, dst_dir)
    svcca_analysis(epochs_layer_dictionary, dst_dir)
if __name__ == '__main__':
    src_dir_path = "../git20200610/examples/asr_swbd/exp/lstm_print_test"
    dst_dir_path = "../git20200610/examples/asr_swbd/exp/lstm_print_test/analysis"
    key = ["convolutions"]
    subkey = ["0"]
    main_fnc(src_dir_path, dst_dir_path, key, subkey)
