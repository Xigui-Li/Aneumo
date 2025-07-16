"""
calWallDistance.py

This script calculates the distance from each internal point to the closest wall point and saves the result 
as a NumPy array with an added SDF (signed distance function) column. Multiple flow rates can be processed
by iterating through a predefined list of flow rate labels.

Usage:
    python calWallDistance.py
"""
import numpy as np

def main(start_id,end_id):
    """
    Process CFD data for case IDs in the range [start_id, end_id). 
    For each case, load the wall and internal arrays, compute distances, 
    and save the resulting SDF data for different flow rates.

    Args:
        start_id (int): Starting case ID (inclusive)
        end_id   (int): Ending case ID (exclusive)
    """
    for case_id in range(start_id,end_id):

        try:
            # Load wall and internal data
            wall_file = 'npydata/m=0.001/array_wall_' + str(case_id) + '.npy'
            internal_file = 'npydata/m=0.001/array_internal_' + str(case_id) + '.npy'

            data_wall = np.load(wall_file)
            data_internal = np.load(internal_file)

            xyz_wall = data_wall[:,0:3]
            xyz_internal = data_internal[:,0:3]
            puvw_internal = data_internal[:,3:]

            # Calculate the distance from each internal point to the closest wall point
            dist = np.sqrt(np.sum((xyz_internal[:, np.newaxis, :] - xyz_wall)**2, axis=2))
            dist_min = np.min(dist,axis=1)
            dist_min = dist_min[:,np.newaxis]

            # Concatenate internal coordinates, minimal distance, and flow fields
            sdf_internal = np.concatenate([xyz_internal,dist_min, puvw_internal],axis=1)

            # Save the SDF array for different flow rates
            for flow_rate in (
                'm=0.001', 'm=0.0015', 'm=0.002', 
                'm=0.0025', 'm=0.003', 'm=0.0035', 
                'm=0.00375', 'm=0.004'
            ):
                sdf_file = 'npydata/' + flow_rate + '/array_sdf_' + str(case_id) + '.npy'
                np.save(sdf_file, sdf_internal)
                print(case_id, flow_rate, 'done')

        except:
            1+1
            
if __name__ == '__main__':
    
    start_id = 3900
    end_id = 4031
    
    main(start_id,end_id)