using HDF5
using Glob

# Copying data excluding large memory MPSs for easier transfer to local machine

for input in glob("*results_L_64_U_8.0*.h5", "./")
output = "stateless_data/" * basename(input)

F = h5open(input,"r")
G = h5open(output,"w")
for s in keys(F)
	if (s != "psi")
		G[s] = read(F,s)
	end
end

close(F)
close(G)

end
