#function that writes the output into lammps format
def writelammps(filename, Lx, Ly, Lz, TIMESTEP, r, v): #r,v are numpy arrays
    #WRITELAMMPS Write data to lammps file
    with open(filename, 'a') as fp:
        ip = len(r)
        fp.write('ITEM: TIMESTEP\n')
        fp.write(f'{TIMESTEP}\n')
        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write(f'{ip}\n') # Nr of atoms
        fp.write('ITEM: BOX BOUNDS pp pp pp\n')
        fp.write(f'{0.0} {Lx}\n') #box size, x
        fp.write(f'{0.0} {Ly}\n')
        fp.write(f'{0.0} {Lz}\n')
        fp.write('ITEM: ATOMS id type x y z vx vy vz\n')
        for i in range(0,ip):
            fp.write(f'{i} {1} {r[i][0]} {r[i][1]} {r[i][2]} {v[i][0]} {v[i][1]} {v[i][2]}\n')