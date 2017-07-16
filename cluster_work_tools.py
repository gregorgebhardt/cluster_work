import argparse
import pandas as pd


def convert_hostfile():
    parser = argparse.ArgumentParser(description='Convert a hostfile.')
    parser.add_argument('infile', type=argparse.FileType('r'),
                        help='the file to convert', metavar="INFILE")
    parser.add_argument('-o', '--outfile', type=argparse.FileType('w'), default=None,
                        help='the file to write the converted list to', metavar="OUTFILE")

    args = parser.parse_args()

    if args.outfile is None:
        args.outfile = open(args.infile.name + '.converted', 'w')

    # count host in hostfile
    print("counting hosts...")
    hosts = dict()
    for line in args.infile:
        line = line.rstrip()
        if line == '':
            continue
        if line in hosts:
            hosts[line] += 1
        else:
            hosts[line] = 1
    args.infile.close()

    print("found {} hosts with {} cpus in total.".format(len(hosts.keys()), sum(hosts.values())))

    for host, cpus in hosts.items():
        line = "{} cpus={}".format(host, cpus)
        print("    ", line)
        args.outfile.write(line + "\n")
    args.outfile.close()


def load_results(path):
    results = pd.read_csv(path, sep='\t')
    results.set_index(keys=['r', 'i'], inplace=True)

    return results
