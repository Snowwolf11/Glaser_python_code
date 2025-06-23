import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import time

from rotation_matrix_algorithm import rotation_matrix



def reparametrize_by_arclength(curve, N):
    """
    Function to reparametrize a given curve to have N equidistant points along its arclength.
    
    Parameters:
    - curve (np.array): An m x 3 array representing the input curve, where m is the number of points on the curve.
    - N (int): The desired number of equidistant points in the output curve.
    
    Returns:
    - curve_repara (np.array): An N x 3 array representing the reparametrized curve.
    - arc_array (np.array): A length-N array where each entry gives the arclength up to the corresponding point in the curve.
    
    Notes:
    - The function uses linear interpolation to calculate the positions of the new points.
    - Works best when m is large, and N > m.
    - TODO: Address the "end problem" to handle the missing points 
    """
    
    diff_curve = np.diff(curve, axis=0)  # Compute difference vectors between consecutive points on the curve
    tot_arc_length = np.cumsum(np.linalg.norm(diff_curve, axis=1), axis=0)[-1]  # Calculate the total arclength of the curve
    arc_array = np.linspace(0, tot_arc_length, N)  # Generate an array of N equidistant arclength values
    arc_diff = arc_array[1] - arc_array[0]  # Spacing between arclength points

    # Initialize the reparametrized curve with the first point of the original curve
    curve_repara_n = curve[0, :]
    curve_repara = [curve_repara_n]
    
    # Indices and flags to track progression through the curve
    m = 0
    i = 1
    i_end = 0
    break_flag = False

    # Loop to iteratively reparametrize the curve
    while (np.linalg.norm(curve[-1, :] - curve_repara_n) > arc_diff or m / len(curve[:, 0]) < 0.9) and (len(curve_repara) <= N):
        # Move forward along the curve until exceeding the required arclength difference
        while np.linalg.norm(curve[m + i, :] - curve_repara_n) < arc_diff:
            i += 1
            if m + i + 1 > len(curve[:, 1]):  # Break if the end of the curve is reached
                break_flag = True
                break
        if break_flag:
            break
        # Identify the segment of the curve where the next point lies
        i_end = i  # i_end is where the next arclength increment is exceeded
                        # --> ||curve(m+i_end) - curve_repara_n|| >= l, ||curve(m+i_end-1) - curve_repara_n|| < l 
                        # --> curve_repara_(n+1) is somewhere on the line (curve(m+i_end) - curve(m+i_end-1))*t + curve(m+i_end-1), t in [0,1]
                        # --> it exists t, so that ||(curve(m+i_end) - curve(m+i_end-1))*t + curve(m+i_end-1) - curve_repara_n|| = l
                        # this problem can be put into the form a*t^2 + b*t + c = 0 and then be solved for t 
                        # --> curve_repara_(n+1) = (curve(m+i_end) - curve(m+i_end-1))*t + curve(m+i_end-1)
                        # the following code implements this calculation to find curve_repara_(n+1)
        a = np.dot(diff_curve[m+i_end-1,:], diff_curve[m+i_end-1,:])
        b = 2*(np.dot(diff_curve[m+i_end-1,:], (curve[m+i_end-1,:] - curve_repara_n)))
        c = np.dot(curve[m+i_end-1,:], curve[m+i_end-1,:]) + np.dot(curve_repara_n, curve_repara_n)  - 2*np.dot(curve[m+i_end-1,:], curve_repara_n) - arc_diff**2
        root_term = b**2 - 4*a*c
        if(root_term < 0):  # Ensure the quadratic equation has real roots
            raise ValueError("root term < 0")
        t = (-b+np.sqrt(root_term))/(2*a)   # Solve for the interpolation parameter
        curve_repara_n = diff_curve[m+i_end-1,:]*t + curve[m+i_end-1,:]     #this is curve_repara_(n+1)
        curve_repara.append(curve_repara_n) # Append the new point to the reparametrized curve
        m = m + i_end -1    # Update the segment start index
        i = 1   # Reset the segment search index

    curve_repara = np.array(curve_repara)   # Convert the list of points to a NumPy array
    
    #TODO at the end there are missing points due to corner cutting of linear interpolation
    delta = np.linalg.norm(curve_repara[-1,:] - curve[-1,:])
    points_missing = N - len(curve_repara[:,1])
    max_offset = max(abs(1-np.linalg.norm(np.diff(curve_repara, axis = 0),axis = 1)/arc_diff))
    #print(f"delta: {delta}, points missing: {points_missing}, maximum offset from arc_diff in %: {max_offset*100}")
    arc_array = arc_array[0:len(curve_repara[:,1])]     # Adjust the arc_array to match the length of the reparametrized curve

    return curve_repara, arc_array

def calculate_curvature_and_torsion(curve, Version = 1):
    """
    Calculates curvature kappa and torsion tau at all points in the curve 
    applying the Frenet Equations.

    Parameters:
    - curve (numpy.ndarray): mx3 array representing the curve, where m is the number of points.
      Points in the curve must be equally spaced.

    Returns:
    - tuple: Two numpy arrays, one for curvature (kappa) and one for torsion (tau).
    """

    if (Version == 1):  #calculates by parametrizing by arclength
        # Ensure the input is a numpy array
        curve = np.asarray(curve)
        if curve.shape[1] != 3:
            raise ValueError("Input curve must be an mx3 numpy array.")

        # Compute the arc length parametrization
        diff_curve = np.diff(curve, axis=0)  # First derivatives of curve
        arc_lengths = np.cumsum(np.linalg.norm(diff_curve, axis=1))
        arc_lengths = np.insert(arc_lengths, 0, 0)  # Include the starting point (0 arc length)

        # Compute derivatives with respect to arc length
        tangent = np.gradient(curve, arc_lengths, axis=0)  # Tangent vector
        tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent_unit = tangent / tangent_norms  # Unit tangent vector

        normal = np.gradient(tangent_unit, arc_lengths, axis=0)  # Normal vector
        normal_norms = np.linalg.norm(normal, axis=1, keepdims=True)
        normal_unit = normal / normal_norms  # Unit normal vector

        binormal = np.cross(tangent_unit, normal_unit)  # Binormal vector

        # Compute curvature (kappa) and torsion (tau)
        curvature = np.linalg.norm(normal, axis=1)
        torsion = -np.einsum('ij,ij->i', np.gradient(binormal, arc_lengths, axis=0), normal_unit)
    elif (Version ==  2):   #TODO doesnt work properly
        curve = np.asarray(curve)
        if curve.ndim != 2 or curve.shape[1] != 3:
            raise ValueError("Input must be an (m x 3) array of 3D points.")

        r_dot = np.gradient(curve, axis=0)
        r_ddot = np.gradient(r_dot, axis=0)
        r_dddot = np.gradient(r_ddot, axis=0)

        cross = np.cross(r_dot, r_ddot)
        cross_norm_sq = np.linalg.norm(cross, axis=1)**2 + 1e-10
        r_dot_norm = np.linalg.norm(r_dot, axis=1)

        # Curvature: |r' × r''| / |r'|^3
        curvature = np.linalg.norm(cross, axis=1) / (r_dot_norm**3 + 1e-10)

        # Torsion: scalar triple product / |r' × r''|^2
        triple = np.einsum('ij,ij->i', np.cross(r_dot, r_ddot), r_dddot)
        eps = 1e-4
        cross_norm_sq = np.clip(cross_norm_sq, eps, None)
        torsion = triple / cross_norm_sq
    return curvature, torsion

def reparametrize_and_calc_curvature_and_torsion(curve, N):
    """reparametrizes by arc_length and calculated curvature and torsion
     Parameters:
    - curve (numpy.ndarray): mx3 array representing the curve, where m is the number of points.
        curve should have more than 10 points and be relatively smooth
    - N (int): (rough) number of points to calculate curve and tors at
    
    Returns:
    - tuple: Two numpy arrays, one for curvature (kappa) and one for torsion (tau).  
    - reparametrized curve as Nx3 np.array (curve_repara), 
    - np.array of lenth N (arc_array), where the i-th entry gives the arclenth upto point i in the curve 
    """
    k = 1.97e-6  # Proportionality constant
    a = 0.815    # Exponent for N
    b = 0.590    # Exponent for curve_length
    estimated_time = k * (N ** a) * (len(curve[:,1]) ** b)
    print(f"Estimated runtime: {estimated_time:.6f} seconds")

    curve_repara, arc_array = reparametrize_by_arclength(curve, N)
    kappa, tau = calculate_curvature_and_torsion(curve_repara)
    return curve_repara, arc_array, kappa, tau

def curve_from_curvature_and_torsion(kappa, tau, delta_s=0.01):        #TODO : test if works correctly
    """calculates curve given curvature and torsion applying the frenet-serret formulas
     Parameters:
    - kappa (ndarray): np.array of length n, containing the values for the curvature along the curve
    - tau (ndarray): np.array of length n, containing the values for the torsion along the curve
    - s: the spacing between the points along the curve (assumed equaly spaced!!)
    
    Returns:
    - curve (numpy.ndarray): mx3 array representing the curve, where m = is the number of points.
    """
    # Initialize tangent (T), normal (N), and binormal (B) vectors
    T = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Initial tangent vector
    N = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # Initial normal vector
    B = np.cross(T, N)  # Initial binormal (cross product of T and N)

    num_points = len(kappa) +1
    # Arrays to store the coordinates of the curve (ensure float64 type)
    curve = np.zeros((num_points, 3), dtype=np.float64)
    r = np.zeros(3, dtype=np.float64)  # Position vector (starts at the origin)
    
    # Integrate the curve
    for i in range(num_points-1):
        # Update vectors using the Frenet-Serret formulas
        dT = kappa[i] * N * delta_s
        dN = (-kappa[i] * T + tau[i] * B) * delta_s
        dB = (-tau[i] * N) * delta_s

        # Numerical integration of the vectors (Euler method)
        T += dT
        N += dN
        B += dB

        # Normalize the vectors
        T /= np.linalg.norm(T)
        N /= np.linalg.norm(N)
        B /= np.linalg.norm(B)

        # Update the position along the curve by integrating the tangent
        r += T * delta_s  # Move in the direction of the tangent

        # Store the current position of the curve
        curve[i+1, :] = r

    return curve

def compare_and_align_curves(curve_1, curve_2, plot = False):
    """
    Translates, scales, rotates curve_2 to align with curve_1 and compares both curves.
    
    Parameters:
        curve_1 (np.ndarray): A 2D array of shape (n, 3) representing the first curve.
        curve_2 (np.ndarray): A 2D array of shape (m, 3) representing the second curve.

    Returns:
        curve_1_final (np.ndarray): Translated, scaled, and rotated version of curve_1.
        curve_2_final (np.ndarray): Translated, scaled, and rotated version of curve_2.
        similarity_score (float): A measure of similarity between the curves (RMSD).
    """
    # Step 1: Check and interpolate if the curves have different lengths
    if len(curve_1) != len(curve_2):
        # Interpolate the shorter curve to match the length of the longer curve
        if len(curve_1) > len(curve_2):
            # Interpolate curve_2 to the length of curve_1
            original_indices = np.linspace(0, 1, len(curve_2))
            target_indices = np.linspace(0, 1, len(curve_1))

            # Create interpolators for each axis
            interp_func_x = interp1d(original_indices, curve_2[:, 0], kind='linear', fill_value="extrapolate")
            interp_func_y = interp1d(original_indices, curve_2[:, 1], kind='linear', fill_value="extrapolate")
            interp_func_z = interp1d(original_indices, curve_2[:, 2], kind='linear', fill_value="extrapolate")

            # Apply the interpolation functions
            curve_2 = np.column_stack([
                interp_func_x(target_indices),
                interp_func_y(target_indices),
                interp_func_z(target_indices)
            ])
        else:
            # Interpolate curve_1 to the length of curve_2
            original_indices = np.linspace(0, 1, len(curve_1))
            target_indices = np.linspace(0, 1, len(curve_2))

            # Create interpolators for each axis
            interp_func_x = interp1d(original_indices, curve_1[:, 0], kind='linear', fill_value="extrapolate")
            interp_func_y = interp1d(original_indices, curve_1[:, 1], kind='linear', fill_value="extrapolate")
            interp_func_z = interp1d(original_indices, curve_1[:, 2], kind='linear', fill_value="extrapolate")

            # Apply the interpolation functions
            curve_1 = np.column_stack([
                interp_func_x(target_indices),
                interp_func_y(target_indices),
                interp_func_z(target_indices)
            ])

    # Step 2: Translate both curves such that their first points are at the origin
    curve_1_translated = curve_1 - curve_1[0]
    curve_2_translated = curve_2 - curve_2[0]

    # Step 3: Scale both curves
    hull_1 = ConvexHull(curve_1)    # Compute the convex hull of the curve
    hull_1_points = curve_1[hull_1.vertices]  # Get the points on the convex hull
    distances_1 = np.linalg.norm(hull_1_points[:, None, :] - hull_1_points[None, :, :], axis=-1)    # Compute pairwise distances between hull points
    scale_factor_1 = np.max(distances_1)
    hull_2 = ConvexHull(curve_2)    # Compute the convex hull of the curve
    hull_2_points = curve_2[hull_2.vertices]  # Get the points on the convex hull
    distances_2 = np.linalg.norm(hull_2_points[:, None, :] - hull_2_points[None, :, :], axis=-1)    # Compute pairwise distances between hull points
    scale_factor_2 = np.max(distances_2)
    curve_1_scaled = curve_1_translated / scale_factor_1
    curve_2_scaled = curve_2_translated / scale_factor_2

    # Step 4V1: Find the best fit rotation
    H = np.dot(curve_1_scaled.T, curve_2_scaled)
    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = np.dot(U, Vt)

    # Step 4V2: rotate initial frames to be equal
    rotation_matrix_V2 = compute_initial_frame_rotation(curve_1_scaled, curve_2_scaled)

    # Apply the rotation to curve_2
    curve_2_rotated = np.dot(curve_2_scaled, rotation_matrix.T)
    curve_2_rotated_V2 = np.dot(curve_2_scaled, rotation_matrix_V2.T)

    # Step 5: Calculate the similarity score (RMSD: Root Mean Squared Distance)
    rmsd = np.sqrt(np.mean(np.sum((curve_1_scaled - curve_2_rotated)**2, axis=1)))

    if plot:
        # Step 6: Plot both curves for comparison
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(curve_1_scaled[:, 0], curve_1_scaled[:, 1], curve_1_scaled[:, 2], label='Curve 1 (Final)', color='blue')
        ax.plot(curve_2_rotated[:, 0], curve_2_rotated[:, 1], curve_2_rotated[:, 2], label='Curve 2 (Aligned)', color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Comparison of Curves After Translation, Scaling, and Rotation (Least Squared Error)')

        plt.show()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(curve_1_scaled[:, 0], curve_1_scaled[:, 1], curve_1_scaled[:, 2], label='Curve 1 (Initial)', color='blue')
        ax.plot(curve_2_rotated_V2[:, 0], curve_2_rotated_V2[:, 1], curve_2_rotated_V2[:, 2], label='Curve 2 (Aligned)', color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Comparison of Curves After Translation, Scaling, and Rotation (Initial Frenet Frames)')

        plt.show()

    # Step 7: Return the final curves and similarity score
    return curve_1_scaled, curve_2_rotated, rmsd

def compute_initial_frame_rotation(curve_1, curve_2, epsilon=1e-6):
    """
    Compute a rotation matrix that aligns the initial Frenet frame of curve_1 to that of curve_2.
    Falls back to aligning only tangents if curvature is near zero.
    Uses user-defined `rotation_matrix(axis, angle)` function.
    
    Parameters:
        curve_1: (N x 3) numpy array
        curve_2: (N x 3) numpy array
        epsilon: threshold to detect degenerate frames
    
    Returns:
        rotation_matrix: (3 x 3) numpy array
    """
    def get_frenet_frame(curve):
        # Tangent vector
        t = curve[1] - curve[0]
        t /= np.linalg.norm(t)

        # Approximate normal vector
        v1 = curve[1] - curve[0]
        v2 = curve[2] - curve[1]
        dv = v2 - v1
        n = dv - np.dot(dv, t) * t

        norm_n = np.linalg.norm(n)
        if norm_n < epsilon:
            return t, None, None  # Flat segment, no usable normal

        n /= norm_n
        b = np.cross(t, n)
        return t, n, b

    def align_vectors(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        if np.allclose(cross, 0):
            if dot > 0:
                return np.eye(3)
            else:
                # 180° rotation around any perpendicular axis
                orth = np.eye(3)[np.argmin(np.abs(v1))]
                axis = np.cross(v1, orth)
                axis /= np.linalg.norm(axis)
                return rotation_matrix(axis, np.pi)

        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        axis = cross / np.linalg.norm(cross)
        return rotation_matrix(axis, angle)

    # Step 1: Get local Frenet frames
    t1, n1, b1 = get_frenet_frame(curve_1)
    t2, n2, b2 = get_frenet_frame(curve_2)

    # Step 2: Decide on alignment strategy
    if n1 is None or n2 is None:
        # Only tangent info available
        return align_vectors(t1, t2)

    # Step 3: Construct rotation from full frame
    frame1 = np.column_stack((t1, n1, b1))
    frame2 = np.column_stack((t2, -n2, -b2))
    return frame2 @ frame1.T

def generate_helical_curve(num_points, radius=1, pitch=1):
    """
    Generates a 3D helical curve.
    
    Parameters:
        num_points (int): Number of points in the curve.
        radius (float): Radius of the helix.
        pitch (float): Pitch of the helix, the height per full rotation.
    
    Returns:
        np.ndarray: 3D coordinates of the helix (shape: [num_points, 3]).
    """
    t = np.linspace(0, 4 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2 * np.pi)
    return np.vstack((x, y, z)).T

def arc_between(p0, v0, p1, v1, num_points=100, tolerance=1e-6):        #TODO: not sure if works properly
    """
    Constructs a circular arc connecting p0 to p1 with tangents v0 and v1.
    Returns sampled points on the arc.
    If the input geometry is not coplanar, returns None and prints a warning.
    """

    # Normalize input directions
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)

    # Check coplanarity: points and tangents must lie in the same plane
    normal = np.cross(v0, v1)
    if np.linalg.norm(normal) < 1e-10:
        # v0 and v1 are parallel → trivial case
        return np.linspace(p0, p1, num_points)

    normal /= np.linalg.norm(normal)

    # Test if p1 lies in the plane defined by (p0, v0, v1)
    to_p1 = p1 - p0
    distance_to_plane = np.dot(to_p1, normal)
    if abs(distance_to_plane) > tolerance:
        print("Warning: Input points and tangents are not coplanar. Cannot construct a circular arc.")
        return None

    # Midpoint and direction along the chord
    m = (p0 + p1) / 2
    u = chord / chord_len

    # Angle between tangents
    angle = np.arccos(np.clip(np.dot(v0, v1), -1, 1))
    radius = chord_len / (2 * np.sin(angle / 2))

    # Circle center lies in plane perpendicular to chord, offset from midpoint
    offset_dir = np.cross(normal, u)
    center = m + np.sqrt(radius**2 - (chord_len / 2)**2) * offset_dir

    # Orthonormal basis for the arc plane
    e1 = (p0 - center) / np.linalg.norm(p0 - center)
    e2 = np.cross(normal, e1)

    # Compute angle range for the arc
    def point(theta): return center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)

    end_vec = (p1 - center)
    theta0 = 0
    theta1 = np.arctan2(np.dot(end_vec, e2), np.dot(end_vec, e1))

    # Ensure arc proceeds in the correct direction
    if np.dot(np.cross(e1, end_vec), normal) < 0:
        theta1 += 2 * np.pi if theta1 < 0 else -2 * np.pi

    theta = np.linspace(theta0, theta1, num_points)
    return np.array([point(t) for t in theta])


############################################################################################################
# tests
def test_reparametrize_by_arclength():  
    t_array = np.linspace(0,2*np.pi*4,1000)
    x_array = np.cos(t_array)
    y_array = np.sin(t_array)
    z_array = t_array/3
    curve = np.transpose(np.array([x_array, y_array, z_array]))
    N = 4000
    curve_repara, arc_array = reparametrize_by_arclength(curve, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])
    ax.plot(curve_repara[:, 0], curve_repara[:, 1], curve_repara[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Curve')
    plt.show()

def test_reparametrize_and_calc_curvature_and_torsion():  
    t_array = np.linspace(0,2*np.pi*4,2000)
    x_array = np.cos(t_array)
    y_array = np.sin(t_array)
    z_array = t_array/3
    curve = np.transpose(np.array([x_array, y_array, z_array]))
    N = 200000
    curve_repara, arc_array, kappa, tau = reparametrize_and_calc_curvature_and_torsion(curve, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])
    ax.plot(curve_repara[:, 0], curve_repara[:, 1], curve_repara[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Curve')
    plt.show()

def test_calculate_curvature_and_torsion():
    """
    Test the calculate_curvature_and_torsion function using a helical curve.
    Compare the calculated curvature and torsion to the expected values.
    Plot the helix and the curvature over the arc length.
    """
    # Define helix parameters
    radius = 1.0
    pitch = 10
    num_points = 500
    t = np.linspace(0, 4 , num_points)

    # Helix curve definition
    x = radius * np.cos(2*np.pi*t)
    y = radius * np.sin(2*np.pi*t)
    z = pitch * t 
    helix = np.vstack((x, y, z)).T

    # Expected curvature and torsion
    k = pitch/(2*np.pi*radius)
    expected_curvature = 1/ (radius*(1 + k**2))
    expected_torsion = k*expected_curvature

    # Calculate curvature and torsion
    calculated_curvature, calculated_torsion = calculate_curvature_and_torsion(helix,2)

    # Plot the helix
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(helix[:, 0], helix[:, 1], helix[:, 2], label='Helix')
    ax.set_title("Helix")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Plot the curvature
    arc_lengths = np.linspace(0, len(helix), len(calculated_curvature))
    ax2 = fig.add_subplot(132)
    ax2.plot(arc_lengths, calculated_curvature, label="Calculated Curvature", linestyle='--')
    ax2.axhline(y=expected_curvature, color='r', linestyle='-', label="Expected Curvature")
    ax2.set_title("Curvature vs Arc Length")
    ax2.set_xlabel("Arc Length")
    ax2.set_ylabel("Curvature")
    ax2.legend()

    # Plot the torsion
    ax3 = fig.add_subplot(133)
    ax3.plot(arc_lengths, calculated_torsion, label="Calculated Torsion", linestyle='--')
    ax3.axhline(y=expected_torsion, color='r', linestyle='-', label="Expected Torsion")
    ax3.set_title("Torsion vs Arc Length")
    ax3.set_xlabel("Arc Length")
    ax3.set_ylabel("Torsion")
    ax3.legend()

    plt.tight_layout()
    plt.show()

# Define the test case for a helix
def test_curve_from_curvature_and_torsion(num_points=1000, r=1.0, c=0.1):
    # Curvature and torsion for a helix
    kappa = np.ones(num_points, dtype=np.float64) / r  # Curvature: 1/r
    tau = np.ones(num_points, dtype=np.float64) * (c / r)  # Torsion: c/r

    # Generate the helix curve
    curve = curve_from_curvature_and_torsion(kappa, tau)
    # Plot the helix
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label='Curve')
    ax.set_title("Curve")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    return curve
    

def test_compare_and_align_curves():
    # Generate identical helices
    curve_1 = generate_helical_curve(100)

    # Apply small translation, scaling, and rotation to curve_1 to create curve_2 (identical curve)
    translation = np.array([2, 3, 1])
    rotation_matrix = R.from_euler('z', np.pi / 4).as_matrix()  # Rotate 45 degrees around Z axis
    curve_2 = (np.dot(curve_1, rotation_matrix.T) + translation)*0.34
    
    print("\nTest 1: Identical Curves (after translation, scaling, and rotation)")
    curve_1_final, curve_2_final, similarity_score = compare_and_align_curves(curve_1, curve_2, plot=True)
    print(f"Similarity Score (RMSD) for identical curves: {similarity_score}")
    
    # Generate different helices (different radius and pitch)
    curve_3 = generate_helical_curve(100, radius=2, pitch=1)
    
    print("\nTest 2: Different Curves (with different radius and pitch)")
    curve_1_final, curve_3_final, similarity_score_diff = compare_and_align_curves(curve_1, curve_3, plot=True)
    print(f"Similarity Score (RMSD) for different curves: {similarity_score_diff}")

def test_arc_between():
    p0 = np.array([1, 0.3, 0])
    v0 = np.array([-2, 0.5, 0])
    p1 = np.array([1, -1, 0])
    v1 = np.array([0.521, 1, 0])

    arc = arc_between(p0, v0, p1, v1)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], label='Circular Arc')
    ax.quiver(*p0, *v0, color='green', label='v0')
    ax.quiver(*p1, *v1, color='red', label='v1')
    ax.scatter(*p0, color='green')
    ax.scatter(*p1, color='red')
    ax.legend()
    plt.title("Smooth Circular Arc Interpolation")
    plt.show()


#test_reparametrize_by_arclength()
#test_calculate_curvature_and_torsion()
#test_curve_from_curvature_and_torsion()
#test_reparametrize_and_calc_curvature_and_torsion()
#test_compare_and_align_curves()
#test_arc_between()

"""
N: 100, curve_length: 1000, time: 0.005784034729003906
N: 200, curve_length: 1000, time: 0.009883880615234375
N: 1000, curve_length: 1000, time: 0.024910688400268555
N: 2000, curve_length: 1000, time: 0.0422205924987793
N: 10000, curve_length: 1000, time: 0.20264482498168945
N: 20000, curve_length: 1000, time: 0.3995380401611328
N: 1000, curve_length: 100, time: 0.02023029327392578
N: 1000, curve_length: 200, time: 0.025844335556030273
N: 1000, curve_length: 1000, time: 0.023782730102539062
N: 1000, curve_length: 2000, time: 0.026135683059692383
N: 1000, curve_length: 10000, time: 0.05770277976989746
N: 1000, curve_length: 20000, time: 0.08845782279968262
N: 1000, curve_length: 200000, time: 0.7382509708404541
N: 1000, curve_length: 2000000, time: 7.877594947814941
N: 200000, curve_length: 2000, time: 4.153922080993652
"""

