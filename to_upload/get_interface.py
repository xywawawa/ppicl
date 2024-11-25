import pymol
from pymol import cmd
import os

pymol.finish_launching(['pymol', '-cq', '--gui=offscreen'])

# Base directory path
base_dir = '/data/a/xiaoyao/pdbs_dataset3/val_data'
save_dir = '/data/a/xiaoyao/dataset3/val_set_interface'

# Function to select interface residues
def select_interface(receptor_chains, ligand_chains, cutoff, output_file):
    """
    根据指定的链和距离，选择界面残基并保存。
    """
    # Select receptor and ligand chains
    pymol.cmd.select('receptor_chains', f'chain {"+".join(receptor_chains)}')
    pymol.cmd.select('ligand_chains', f'chain {"+".join(ligand_chains)}')
    
    # Select interface based on residues
    pymol.cmd.select('interface', f'byres (receptor_chains within {cutoff} of ligand_chains) or byres (ligand_chains within {cutoff} of receptor_chains)')
    
    # Save the interface to a new PDB file
    pymol.cmd.save(output_file, 'interface')

# Iterate through each subdirectory
for subdir in os.listdir(base_dir):
    pdb_dir = os.path.join(base_dir, subdir)
    output_dir = os.path.join(save_dir, subdir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for pdb_file in os.listdir(pdb_dir):
        try:
        # Ensure to delete all objects from PyMOL before loading new structure
            pymol.cmd.delete('all')
            
            # Load the PDB file from the full path
            full_pdb_path = os.path.join(pdb_dir, pdb_file)
            pymol.cmd.load(full_pdb_path, 'complex', quiet=1)
            
            # Define output path
            output_file = os.path.join(output_dir, pdb_file)
            
            # Extract receptor and ligand chains from the subdir name
            receptor_chains = subdir.split('_')[1][0]
            ligand_chains = subdir.split('_')[1][1]
            
            # Select and save the interface residues
            select_interface(receptor_chains, ligand_chains, 6.0, output_file)
            
            # Print completion message for each file
            print(f"Completed: {output_file}\n")
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

# Quit PyMOL after all processing is complete
print("Interface extraction completed.")
pymol.cmd.quit()