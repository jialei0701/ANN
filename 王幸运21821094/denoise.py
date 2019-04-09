#!/usr/bin/env python3

import argparse
import denoiser
import os
import sys




parser = argparse.ArgumentParser(description='Denoising network.')
parser.add_argument('--inmodel', metavar='N', type=str, nargs=1,
                   help='Stored model of the network')


parser.add_argument('--mode', metavar='N', choices=['train', 'test'], nargs=1,
                   help='Network mode', default=['train'])

parser.add_argument('--outmodel', metavar='N', type=str, nargs=1,
                   help='Save location for the model of the network')

parser.add_argument('--network_type', metavar='N', type=str, nargs=1,
                   help='Network type options', default=[''])

parser.add_argument('--apply', action='store_true',
                   help='Apply to a single image')




parser.add_argument('--gamma', metavar='N', type=float, nargs=1,
                   help='Gamma correct', default=[.454545])

parser.add_argument('--out_dir', metavar='N', type=str, nargs=1,
                   help='Output directory')

parser.add_argument('--renders_dir', metavar='N', type=str, nargs=1,
                   help='Dataset dir')



parser.add_argument('--novis',  action='store_true')


parser.add_argument('--safe_border', metavar='N', type=int, nargs=1, default=[22],
                   help='Sets save border')



parser.add_argument('--spp_add', metavar='N', type=int, nargs=1,
                   help='spp_additional', default=[3])

parser.add_argument('--max_epoch', metavar='N', type=int, nargs=1,
                   help='Max epochs', default=[70000000])



parser.add_argument('--mapmode', metavar='N', type=str, nargs=1,
                   help='Map mode', default=[None])


parser.add_argument('--uniform',  action='store_true',
                   help='set uniform for apply')



parser.add_argument('--filter', metavar='N', type=str, nargs=1,
                   help='filter dataset', default=[None])

parser.add_argument('--dualmode',  metavar='N',  type=int, nargs=1, default=False,
                   help='Mode of training')

args = parser.parse_args()


renders_dir = None
if args.renders_dir is not None:
    renders_dir = args.renders_dir[0]



denoising = denoiser.Denoising(network_type=args.network_type[0],
                                       spp=4,
                                       renders_dir=renders_dir,
                               spp_additional=args.spp_add[0], 
                               mapmode=args.mapmode[0],
                               uniform=args.uniform,
                               dataset_filter=args.filter[0],
                                dualmode=args.dualmode
                                )


if args.novis:
    denoising.set_vis(False)

if args.gamma is not None:
    denoising.set_gamma(args.gamma[0])
elif args.apply:
    denoising.set_gamma(args.gamma[0])

if args.out_dir is not None:
    denoising.set_out_dir(args.out_dir[0])


if args.safe_border is not None:
    denoising.set_save_border(args.safe_border[0])




if args.inmodel is not None:
    with open(os.path.abspath(args.inmodel[0]), 'rb') as f:
        denoising.load_model(f)



if args.outmodel is not None:
	denoising.set_save_model_path(args.outmodel[0])





if args.mode is not None:
    denoising.set_mode(args.mode[0])


if args.apply:

    denoising.apply()
else:
    if args.mode[0] == 'train':
        denoising.train(args.max_epoch[0])

