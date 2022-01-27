## json builder
## Md Asif Jalal 2021 
## The University of Sheffield


import os
import json
import argparse


def json_builder(in_filename, out_filename, batch_size=40, median_frame_length=400):
    out_dict = {}
    with open(in_filename, "r") as jsonfile:
        data = json.load(jsonfile)
        #data=sorted(data.items(), key = lambda x: x[1]['utt2num_frames']) 
        counter = 0
        while counter<=batch_size:
            pop_item = min(data.items(), key = lambda x: abs(x[1]['utt2num_frames']-median_frame_length))
            key,_ = pop_item
            value=data.pop(key)
            out_dict[key]=value
            counter+=1
            
    with open(out_filename, "w") as jsonfile:
        json.dump(out_dict, jsonfile, indent=4)
    return 0

def main(parser):
    args = parser.parse_args()
    json_builder(args.input,args.output,args.batch_size,args.sample_length)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='json file builder analysis espresso')
    parser.add_argument('-i', '--input', action="store")
    parser.add_argument('-o', '--output', action="store")
    parser.add_argument('-b','--batch_size', action="store", type=int, default=40)
    parser.add_argument('-s','--sample_length', action="store", type=int, default=400)
    main(parser)
