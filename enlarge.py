#!/bin/python3
import pandas as pd
import math
import sys
import argparse
import json
import numpy as np
import concurrent.futures
from sklearn.cluster import KMeans
from collections import Counter

########################################################################################################
############################################# INFRASTRUCTURE ###########################################
########################################################################################################

a_no_to_symbol = {1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne",
    11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar", 19:"K", 20:"Ca",
    21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn",
    31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y", 40:"Zr",
    41:"Nb", 42:"Mo", 43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn",
    51:"Sb", 52:"Te", 53:"I", 54:"Xe", 55:"Cs", 56:"Ba", 57:"La", 58:"Ce", 59:"Pr", 60:"Nd",
    61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy", 67:"Ho", 68:"Er", 69:"Tm", 70:"Yb",
    71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
    81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra", 89:"Ac", 90:"Th",
    91:"Pa", 92:"U", 93:"Np", 94:"Pu", 95:"Am", 96:"Cm", 97:"Bk", 98:"Cf", 99:"Es", 100:"Fm",
    101:"Md", 102:"No", 103:"Lr", 104:"Rf", 105:"Db", 106:"Sg", 107:"Bh", 108:"Hs", 109:"Mt", 110:"Ds",
    111:"Rg"}

symbol_to_a_no = {}
for a_no in a_no_to_symbol:
    symbol_to_a_no[a_no_to_symbol[a_no]] = a_no


def cart(x,y):
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)**0.5
    
class Atom:
    def __init__(self, coord, a_no):
        self.coord = coord
        self.a_no = a_no   
        
class Fragment:
    def __init__(self, atoms = []):
        self.atoms = []
        self.charge = 0
        self.weighted_centre = [0.0,0.0,0.0]
        for atom in atoms:
            self.add_atom(atom)
            
    def add_atom(self, atom):
        self.atoms.append(atom)
        self.charge += atom.a_no
        for i in range(3):
            self.weighted_centre[i] += atom.coord[i]*atom.a_no
            
    def centre_of_charge(self):
        return [self.weighted_centre[i]/self.charge for i in range(3)]
    
    def locality(self):
        centre = self.centre_of_charge()
        dist = 0.0
        for atom in self.atoms:
            dist += cart(atom.coord,centre)**2
        dist /= len(self.atoms)
        return dist
    
    def fingerprint(self):
        elements = {}
        for atom in self.atoms:
            symbol = a_no_to_symbol[atom.a_no]
            if symbol not in elements:
                elements[symbol] = 0
            elements[symbol] += 1
            
        el_list = list(elements.items())
        el_list.sort(key=lambda x:x[0])
        
        
        ret = ""
        for el in el_list:
            ret += el[0]
            ret += str(el[1])
        return ret
            
def union(fragments):
    new_frag = Fragment([])
    for frag in fragments:
        new_frag.atoms.extend(frag.atoms)
        new_frag.charge += frag.charge
        for i in range(3):
            new_frag.weighted_centre[i] += frag.weighted_centre[i]
    return new_frag


# Represents topology component of input json
class Topology:
    def __init__(self):
        self.atoms = []
        self.connectivity = []
        self.fragments = []
        self.fragment_formal_charges = []
        self.xyz = None
        
    def assemble_fragments(self):
        ret = []
        for atom_ixs in self.fragments:
            ret.append(Fragment([self.atoms[i] for i in atom_ixs]))
        return ret
        
    def from_xyz(self, xyz_file):
        raw_symbols = []
        raw_coords = []

        self.xyz = xyz_file
        with open(self.xyz) as xyz:
            xyz.readline()
            xyz.readline()
            for line in xyz.readlines():
                dat = line.split()
                raw_symbols.append(dat[0])
                raw_coords.append(float(dat[1]))
                raw_coords.append(float(dat[2]))
                raw_coords.append(float(dat[3]))

        coords = []
        symbols = []
        skip = 0
        for i in range(0, len(raw_coords), 3):
            #if skip > 0:
            #    skip -= 1
            #    continue
            #if raw_coords[i] < 0.0:
            #    skip = 2
            #    continue
            coords.append([raw_coords[i],raw_coords[i+1],raw_coords[i+2]])
            symbols.append(raw_symbols[i//3])
        print(Counter(symbols))

        for (symbol, coord) in zip(symbols,coords):
            self.atoms.append(Atom(coord, symbol_to_a_no[symbol]))

        if len(symbols)%3 != 0:
            print("GIVE WATER ONLY")


        self.connectivity = []
        self.fragments = []
        for i in range(0,len(symbols),3):
            if symbols[i] != 'O' or symbols[i+1] != 'H' or symbols[i+2] != 'H':
                raise "WATER WATER WATER"
            self.fragments.append([i,i+1,i+2])
        self.fragment_formal_charges = [0 for _ in range(len(self.fragments))]

    def from_json(self, topology_json, connected_fragments = False):

        symbols = []
        raw_coords = []
        if 'symbols' in topology_json and 'geometry' in topology_json:
            symbols = topology_json['symbols']
            raw_coords = topology_json['geometry']


        elif 'xyz' in topology_json:
            self.xyz = topology_json['xyz']
            with open(self.xyz) as xyz:
                xyz.readline()
                xyz.readline()
                for line in xyz.readlines():
                    dat = line.split()
                    symbols.append(dat[0])
                    raw_coords.append(float(dat[1]))
                    raw_coords.append(float(dat[2]))
                    raw_coords.append(float(dat[3]))
        else:
            raise "Bad input"

        coords = []
        for i in range(0, len(raw_coords), 3):
            coords.append([raw_coords[i],raw_coords[i+1],raw_coords[i+2]])

        for (symbol, coord) in zip(symbols,coords):
            self.atoms.append(Atom(coord, symbol_to_a_no[symbol]))

        nfrag = len(topology_json['fragments'])


        if 'connectivity' in topology_json:
            self.connectivity = topology_json['connectivity']
        else:
            self.connectivity = []

        if connected_fragments:
            self.fragment_by_connectivity()
            self.fragment_formal_charges = [0 for _ in range(len(self.fragments))]
            print("FORMAL CHARGES DON'T WORK WITH CONNECTED FRAGMENTATION")
        else:
            self.fragments = topology_json['fragments']
            self.fragment_formal_charges = topology_json["fragment_formal_charges"]

        
    def fragment_by_connectivity(self):
        
        reps = [i for i in range(len(self.atoms))]
        for (i,j,_) in self.connectivity:
            x = 0
            y = 0
            if reps[i] < reps[j]:
                x,y = i,j
            else:
                x,y = j,i
            component = []
            to_change = reps[y]
            for k in range(len(reps)):
                if reps[k] == to_change:
                    reps[k] = reps[x]
                if reps[k] == reps[x]:
                    component.append(k)
                    
        i = 0
        frags = [[] for _ in set(reps)]
        for rep in set(reps):
            for j in range(len(self.atoms)):
                if reps[j] == rep:
                    frags[i].append(j)
            i += 1
        self.fragments = frags
        
    def to_json(self):
        topology_json = {}
        topology_json['fragments'] = self.fragments
        topology_json['fragment_formal_charges'] = self.fragment_formal_charges
        if self.connectivity:
            topology_json['connectivity'] = self.connectivity
        
        if self.xyz == None:
            topology_json['geometry'] = []
            topology_json['symbols'] = []
            for atom in self.atoms:
                topology_json['symbols'].append(a_no_to_symbol[atom.a_no])
                topology_json['geometry'].extend(atom.coord)
        else:
            topology_json['xyz'] = self.xyz
        return topology_json

    def nfrag(self):
        return len(self.fragments)
    
    def natoms(self):
        return len(self.atoms)
    
    def group_fragments(self, group_map):
        assert(len(group_map) == len(self.fragments))
        ret = Topology()
        ret.atoms = self.atoms
        ret.connectivity = self.connectivity
        ret.xyz = self.xyz
        
        nfrag = max(group_map)+1
        ret.fragments = [[] for _ in range(nfrag)]
        ret.fragment_formal_charges = [0 for _ in range(nfrag)]
        for old_frag_ix, new_frag_ix in enumerate(group_map):
            ret.fragments[new_frag_ix].extend(self.fragments[old_frag_ix])
            ret.fragment_formal_charges[new_frag_ix] += self.fragment_formal_charges[old_frag_ix]
        return ret

def fingerprint_fragments(frags):
    fingerprints = {}
    for frag in frags:
        prnt = frag.fingerprint()
        if prnt not in fingerprints:
            fingerprints[prnt] = 0
        fingerprints[prnt] += 1
    return fingerprints

def check_for_pairs(frags):
    fingerprints = fingerprint_fragments(frags)

    if len(set(fingerprints.keys())) == 2 and len(set(fingerprints.values())) == 1 :
        return list(fingerprints.keys())
    else:
        return None

def pair_fragments(frags, fragsA_map, fragsB_map):    
    membership = []
    fragsA = [frags[ix] for ix in fragsA_map]
    fragsB = [frags[ix] for ix in fragsB_map]
    for i in range(len(fragsB)):
        membership.append(i)
            
    def gain(i,j):
        ci = fragsA[membership[i]].centre_of_charge()
        cj = fragsA[membership[j]].centre_of_charge()
        current = cart(fragsB[i].centre_of_charge(),ci) + cart(fragsB[j].centre_of_charge(),cj)
        swapped = cart(fragsB[j].centre_of_charge(),ci) + cart(fragsB[i].centre_of_charge(),cj)
        return current - swapped
    
    prev_mem = []
    for i in range(20):
        if i > 0:
            done = True
            for x,y in zip(membership, prev_mem):
                if x != y:
                    done = False
                    break
            if done:
                print("CONVERGED!")
                break
        prev_mem = membership.copy()
        print("Iteration",i)

        deltas = []
        for i,frag in enumerate(fragsB):
            mem = membership[i]
            clust = -1
            best_alternative = 1000000.0
            for j,centroid in [(j,fragsA[j].centre_of_charge()) for j in range(len(fragsA)) if j != mem]:
                dist = cart(frag.centre_of_charge(),centroid)
                if dist < best_alternative:
                    best_alternative = dist
                    clust = j
            deltas.append((i,best_alternative-cart(frag.centre_of_charge(),fragsA[mem].centre_of_charge()),clust))

        deltas = sorted(deltas, key=lambda x: x[1])

        for (i,delta,best) in deltas:
            for (j,fragj) in enumerate(fragsA):
                if membership[i] == j:
                    continue
                done = False
                if gain(i,j) > 0:
                    membership[i], membership[j] = membership[j], membership[i]
                    done = True
                if done:
                    break
    
    ret = []
    for i in range(len(frags)):
        if i in fragsA_map:
            for j in range(len(fragsA_map)):
                if fragsA_map[j] == i:
                    ret.append(j)
                    break
        else:
            for j in range(len(fragsB_map)):
                if fragsB_map[j] == i:
                    ret.append(membership[j])
                    break
    return ret


#def cluster_fragments(frags, multiplicity):
#    centres = np.array([frag.centre_of_charge() for frag in frags])
#    weights = np.array([frag.charge for frag in frags])
#    
#    clusters = len(frags)//multiplicity
#    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(centres, sample_weight=weights)
#    print(Counter(kmeans.labels_))
#    
#    print("Equalizing")
#    return equalize_clusters(frags, kmeans.labels_, kmeans.cluster_centers_)

def subcluster(p):
    i,frags,centres,weights,clusters,multiplicity = p
    print("Group",i,"size",len(frags),"clusters",clusters)
    kmeans = KMeans(n_clusters=clusters, random_state=0,n_init=10).fit(centres, sample_weight=weights)
    x = equalize_clusters(frags, kmeans.labels_, kmeans.cluster_centers_, multiplicity)
    print("Group",i,"done")
    return x

def cluster_fragments(frags, multiplicity):
    centres = np.array([frag.centre_of_charge() for frag in frags])
    weights = np.array([frag.charge for frag in frags])
    
    frags_per_cluster = 500
    pre_clusters = len(frags)//frags_per_cluster

    print("Pre-clustering into",pre_clusters,"groups")
    pc_kmeans = KMeans(n_clusters=pre_clusters, random_state=0,n_init=10).fit(centres, sample_weight=weights)
    labels = pc_kmeans.labels_
    cluster_map = [-1 for _ in range(len(frags))]
    with concurrent.futures.ProcessPoolExecutor() as pool:
        problem_list = []
        counts = Counter(labels)
        for i in range(pre_clusters):
        #for i in range(10):
            sub_frags = [frags[j] for (j,lab) in enumerate(labels) if lab == i]
            sub_centres = [centres[j] for (j,lab) in enumerate(labels) if lab == i]
            sub_weights = [weights[j] for (j,lab) in enumerate(labels) if lab == i]
            nfrag = counts[i]
            problem_list.append((i,sub_frags,sub_centres,sub_weights,
                math.ceil(nfrag/multiplicity),multiplicity))
        out = pool.map(subcluster, problem_list)

        cluster_ix = 0
        for (i,o) in enumerate(out):
            print(i,"Done, ix =",cluster_ix,"to",cluster_ix+max(o)+1)
            # print(Counter(o))
            frag_global_ixs = [j for (j,lab) in enumerate(labels) if lab == i]
            for (frag_ix,cluster) in enumerate(o):
                global_ix = frag_global_ixs[frag_ix]
                cluster_map[global_ix] = cluster+cluster_ix
            cluster_ix += max(o)+1
    print(Counter(Counter(cluster_map).values()))
    return cluster_map            

    # counts = Counter(pc_kmeans.labels_)
    # for i in range(pre_clusters):
    #     sub_frags = [frags[j] for (j,lab) in enumerate(pc_kmeans.labels_) if lab == i]
    #     sub_centres = [centres[j] for (j,lab) in enumerate(pc_kmeans.labels_) if lab == i]
    #     sub_weights = [weights[j] for (j,lab) in enumerate(pc_kmeans.labels_) if lab == i]
    #     nfrag = counts[i]
    #     clusters = math.ceil(nfrag/multiplicity)
    #     print("Group",i,"size",nfrag,"clusters",clusters)
    #     kmeans = KMeans(n_clusters=clusters, random_state=0).fit(sub_centres, sample_weight=sub_weights)
    #     x = equalize_clusters(sub_frags, kmeans.labels_, kmeans.cluster_centers_, multiplicity)
    #     print(x)

    #clusters = len(frags)//multiplicity
    #kmeans = KMeans(n_clusters=clusters, random_state=0).fit(centres, sample_weight=weights)
    #print(Counter(kmeans.labels_))
    #
    #print("Equalizing")
    #return equalize_clusters(frags, kmeans.labels_, kmeans.cluster_centers_)

def equalize_clusters(frags, membership, centroids, multiplicity=None):
    # make clusters equal sizes
    def gain(i,j):
        ci = centroids[membership[i]]
        cj = centroids[membership[j]]
        current = cart(frags[i].centre_of_charge(),ci) + cart(frags[j].centre_of_charge(),cj)
        swapped = cart(frags[j].centre_of_charge(),ci) + cart(frags[i].centre_of_charge(),cj)
        return current - swapped
    
    def fragsize(i):
        return sum([1 if mem == i else 0 for mem in membership])
    
    if multiplicity == None:
        multiplicity = math.ceil(len(frags)/len(centroids)) 

    #if len(frags)%multiplicity < multiplicity/2:
    #    multiplicity -= 1

    prev_mem = []
    for iteration in range(100):
        if iteration == 99:
            raise "FAILED TO CONVERGE"
        
        if iteration > 0:
            done = True
            for x,y in zip(membership, prev_mem):
                if x != y:
                    done = False
                    break
            if done:
                #print("CONVERGED!")
                break
        prev_mem = membership.copy()
        # Calculate centroids
        #print("ITER")
        centroids = [[0.0,0.0,0.0] for _ in range(len(centroids))]
        counts = [0 for _ in range(len(centroids))]

        for (j,mem) in enumerate(membership):
            centroids[mem][0] += frags[j].centre_of_charge()[0]*frags[j].charge
            centroids[mem][1] += frags[j].centre_of_charge()[1]*frags[j].charge
            centroids[mem][2] += frags[j].centre_of_charge()[2]*frags[j].charge
            counts[mem] += frags[j].charge

        for i in range(len(centroids)):
            if counts[i] > 0:
                centroids[i][0] /= counts[i]
                centroids[i][1] /= counts[i]
                centroids[i][2] /= counts[i]

        # Old centroid calculation
        #for i in range(len(centroids)):
        #    cent = [0.0,0.0,0.0]
        #    n = 0
        #    for (j,mem) in enumerate(membership):
        #        if mem == i:
        #            cent[0] += frags[j].centre_of_charge()[0]*frags[j].charge
        #            cent[1] += frags[j].centre_of_charge()[1]*frags[j].charge
        #            cent[2] += frags[j].centre_of_charge()[2]*frags[j].charge
        #            n += frags[j].charge
        #    if n > 0:
        #        cent[0] /= n
        #        cent[1] /= n
        #        cent[2] /= n
        #    centroids[i] = cent
        
        outgoing = {}
        for i in range(len(centroids)):
            outgoing[i] = []

        deltas = [None for _ in range(len(frags))]
        for i,frag in enumerate(frags):
            mem = membership[i]
            clust = -1
            best_alternative = 1000000.0

            coc = frag.centre_of_charge()
            distances = map(lambda x: (x[0],cart(coc, x[1])), filter(lambda x: x[0] != mem, enumerate(centroids)))
            best = min(distances, key=lambda x: x[1])

            clust = best[0]
            best_alternative = best[1]

            #for j,centroid in [(j,centroids[j]) for j in range(len(centroids)) if j != mem]:
            #    dist = cart(frag.centre_of_charge(),centroid)
            #    if dist < best_alternative:
            #        best_alternative = dist
            #        clust = j
            deltas[i] = (i,best_alternative-cart(frag.centre_of_charge(),centroids[membership[i]]),clust)

            # best_alternative = distance to nearest other centroid
            # d = distance to current centroid
            # delta is negative iff the best alternative is closer

        deltas = sorted(deltas, key=lambda x: x[1], reverse=False)

        for (i,delta,best) in deltas:
            # Consider only closest N fragments?
            #closest_frags = sorted(enumerate(centroids), key=lambda x: cart(x[1], centroids[membership[i]]))[:10]
            for (j,centroid) in enumerate(centroids):
                if membership[i] == j:
                    continue

                done = False
                for k in range(len(outgoing[j])):
                    if gain(i,outgoing[j][k]) > 0:
                        swap = outgoing[j].pop(k)
                        #print("Swap", i, "with", swap, "for gain")
                        membership[i], membership[swap] = membership[swap], membership[i]
                        done = True
                        break
                if done:
                    break

                jsize = fragsize(j)
                isize = fragsize(membership[i])

                # If i is too big and j is too small, transfer
                if jsize < multiplicity < isize:
                    #print("Move frag", i, "to cluster", j, "for size")
                    membership[i] = j
                    break
                # If i,j both too small, transfer smaller to bigger
                if isize < jsize < multiplicity:
                    membership[i] = j
                    break
                if jsize < multiplicity-1 and isize == multiplicity:
                    membership[i] = j
                    break
                    
                outgoing[membership[i]].append(i)

        localities = [0.0 for _ in range(len(centroids))]
        for (frag,mem) in enumerate(membership):
            localities[mem] += cart(frags[frag].centre_of_charge(), centroids[mem])

        #print("Iteration",iteration,"Goodness",sum(localities))
        #print(Counter(membership))

    return membership

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enlarge fragments of HERMES topology by combining nearby fragments.')
    parser.add_argument('--pair', help='Pair fragments before clustering.', action='store_true')
    parser.add_argument('--infer-fragments', help='Infer fragmentation from connectivity.', action='store_true')
    parser.add_argument('--multiplicity', type=int, action='store', help='Number of fragments per group.', required=True)
    parser.add_argument('input', type=str, action='store', help='Path to input json.')
    parser.add_argument('-o', type=str, action='store', help='Path to output file.', default='output.json')
    
    args = parser.parse_args()
    return args

########################################################################################################
############################################## MAIN METHODS ############################################
########################################################################################################

def form_paired_topology(topology):
    fragments = topology.assemble_fragments()
    pair_fingerprints = check_for_pairs(fragments)
    print("Pairing", pair_fingerprints[0], "with", pair_fingerprints[1])
    fragsA = []
    fragsB = []
    for (i,frag) in enumerate(fragments):
        if frag.fingerprint() == pair_fingerprints[0]:
            fragsA.append(i)
        else:
            fragsB.append(i)

    paired_fragments_map = pair_fragments(fragments,fragsA,fragsB)
    paired_topology = topology.group_fragments(paired_fragments_map)
    return paired_topology

def form_clustered_topology(topology, multiplicity):
    fragments = topology.assemble_fragments()
    print("Clustering")
    enlarged_fragments_map = cluster_fragments(fragments, multiplicity)
    print("Grouping")
    enlarged_topology = topology.group_fragments(enlarged_fragments_map)
    return enlarged_topology

def main():
    args = parse_arguments()

    topology = Topology()

    extension = args.input.split('.')[1]
    if extension == 'json':
        with open(args.input) as f:
            input_file = json.load(f)
            topology.from_json(input_file, connected_fragments=args.infer_fragments)
    elif extension == 'xyz':
        print("THIS BETTER BE WATER")
        topology.from_xyz(args.input)
    
    print("Initial fragment count:", topology.nfrag())

    #max_d = 0.0
    #avg_d = 0.0
    #total_d = 0
    #ds = []
    #for frag in topology.fragments:
    #    atoms = [topology.atoms[at] for at in frag]
    #    fragment = Fragment(atoms)
    #    coc = fragment.centre_of_charge()
    #    for at in atoms:
    #        if cart(coc,at.coord) > 20.0:
    #            print(at.coord)
    #            local_ds = []
    #            for other in topology.atoms:
    #                local_ds.append(cart(at.coord, other.coord))
    #            print(pd.DataFrame(local_ds).describe())
    #            print(pd.DataFrame(local_ds).quantile(0.05))
    #            print()
    #        ds.append(cart(coc,at.coord))
    #        max_d = max(max_d, cart(coc, at.coord))
    #        avg_d += cart(coc,at.coord)
    #        total_d += 1
    #df = pd.DataFrame(ds)
    #print(df.describe())
    #print("max_d =",max_d)
    #print("avg_d =",avg_d/total_d)
    #return 0

    # Check if input can be paired
    paired_input = check_for_pairs(topology.assemble_fragments())
    if args.pair and not paired_input:
        fingerprints = fingerprint_fragments(topology.assemble_fragments())
        print("Fragment pairing requested, but no pairs detected. Fragment fingerprints:")
        for key in fingerprints:
            print(key, fingerprints[key])
        sys.exit()
    elif paired_input:
        print("NOTE: Input fragments are paired. Did you mean to add the --pair option?")

    # Pair fragments
    if args.pair:
        print("Pairing fragments")
        topology = form_paired_topology(topology)
        print("Paired fragment count:", topology.nfrag())

    print("Clustering fragments with multiplicity", args.multiplicity)
    topology = form_clustered_topology(topology,args.multiplicity)
    print("Final fragment count:", topology.nfrag())

    max_d = 0.0
    avg_d = 0.0
    total_d = 0
    ds = []
    for frag in topology.fragments:
        atoms = [topology.atoms[at] for at in frag]
        fragment = Fragment(atoms)
        coc = fragment.centre_of_charge()
        for at in atoms:
            d = cart(coc,at.coord)
            ds.append(cart(coc,at.coord))
            max_d = max(max_d, cart(coc, at.coord))
            avg_d += cart(coc,at.coord)
            total_d += 1
    df = pd.DataFrame(ds)
    print(df.describe())
    print("max_d =",max_d)
    print("avg_d =",avg_d/total_d)

    with open(args.o,'w') as f:
        json.dump(topology.to_json(),f,indent=4)

if __name__ == "__main__":
    main()
