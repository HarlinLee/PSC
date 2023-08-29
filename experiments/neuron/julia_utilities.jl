module  julia_utilities

using Plots
using Plots.PlotMeasures
using LinearAlgebra
using Distributions
using Random

export 
    find_preimage_rows,
    plot_preimage_on_torus,
    plot_preimage_on_torus_all_fields,
    plot_preimage_on_torus_with_fields,
    plot_distance_overlap_correlation,
    vector_to_symmetric_matrix,
    calculate_circle_distance,
    generate_uniform_neurons_on_circle,
    generate_circular_path,
    apply_relu_to_matrix,
    linear_relu_response_function,
    calculate_neural_response_matrix,
    generate_random_circle_walk,
    generate_random_skipping_circular_walk,
    calculate_output_neural_response_matrix,
    add_normal_random_noise,
    calculate_nonuniform_neural_response_matrix,
    generate_random_bounded_circle_walk



function vector_to_symmetric_matrix(vector, n)
    D = zeros(n,n)
    lower_index = [j+(i-1)*n  for i=1:n for j=i+1:n]
    D[lower_index] = vector
    D = D + transpose(D)
    return D
    
end



function find_preimage_rows(connection_matrix, column_index, threshold)
    preimage_rows = []
    for row_index in axes(connection_matrix)[1]
        if connection_matrix[row_index, column_index] >= threshold
            append!(preimage_rows, row_index)
        end
    end
    return preimage_rows
end


# Connection matrix of dimensions (# reg 1 neurons, # reg 2 neurons)
function plot_preimage_on_circle_all_fields(connection_matrix::Matrix{Float64}, neurons_on_circle::Vector{Float64}, column_index::Int64; title = "")
    scatter(title = title)
    for row_index in axes(connection_matrix)[1]
        strength = connection_matrix[row_index, column_index]
        if strength >= 0
            scatter!((neurons_on_circle[row_index], 0), color = cgrad([:white, :red])[strength], markerstrokecolor = :white, markersize = 3, legend = false)
        else
            scatter!((neurons_on_circle[row_index], 0), color = cgrad([:black, :white])[strength+1], markerstrokecolor = :white, markersize = 3, legend = false)
        end
    end
    scatter!(ylims = (-.1,0.1))
end

function plot_preimage_on_torus_all_fields(connection_matrix, neurons_on_torus, column_index, title::String = nothing)
    figure = scatter(title = title)
    for row_index in axes(connection_matrix)[1]
        strength = connection_matrix[row_index, column_index]
        if strength >= 0
            scatter!((neurons_on_torus[1, row_index], neurons_on_torus[2, row_index]), color = cgrad([:white, :red])[strength], markerstrokecolor = :white, legend = false)
        else
            scatter!((neurons_on_torus[1, row_index], neurons_on_torus[2, row_index]), color = cgrad([:black, :white])[strength+1], markerstrokecolor = :white, legend = false)
        end
    end
    figure
end



function plot_preimage_on_torus(neurons_on_torus, row_indices)
    all_neurons = Array{Float64}(undef, 0,2)
    for index in axes(neurons_on_torus)[2]
        x_coord = neurons_on_torus[1, index]
        y_coord = neurons_on_torus[2, index]
        all_neurons = [all_neurons; [x_coord y_coord]]
    end
    preimage_neurons = Array{Float64}(undef, 0, 2)
    for row_index in row_indices
        x_coord = neurons_on_torus[1, row_index]
        y_coord = neurons_on_torus[2, row_index]
        preimage_neurons = [preimage_neurons; [x_coord y_coord]]
    end
    scatter(all_neurons[:,1], all_neurons[:,2], xlims = (-0.1,1.1), ylims = (-0.1,1.1), color = :black, markersize = 1, legend = false)
    scatter!(preimage_neurons[:,1], preimage_neurons[:,2], color = :red, markersize = 3)
end



function circleShape(h, k, r)
    θ = LinRange(0, 2*π, 500)
    h .+ r*sin.(θ), k .+ r*cos.(θ)
end



function plot_preimage_on_torus_with_fields(neurons_on_torus, row_indices, radius)
    all_neurons = Array{Float64}(undef, 0,2)
    for index in axes(neurons_on_torus)[2]
        x_coord = neurons_on_torus[1, index]
        y_coord = neurons_on_torus[2, index]
        all_neurons = [all_neurons; [x_coord y_coord]]
    end
    plot = scatter(all_neurons[:,1], all_neurons[:,2], xlims = (-0.1,1.1), ylims = (-0.1,1.1), color = :black, markersize = 1, legend = false)
    for row_index in row_indices
        x_coord = neurons_on_torus[1, row_index]
        y_coord = neurons_on_torus[2, row_index]
        scatter!((x_coord, y_coord), color = :red, markersize = 3)
        plot!(circleShape(x_coord, y_coord, radius), seriestype = [:shape], c = :grey, linecolor = false, legend = false, fillalpha = 0.05)
    end
    plot
end



function plot_distance_overlap_correlation(connection_matrix, distance_matrix, fixed_output_index)
    points_to_plot = Array{Float64}(undef, 0, 2)
    for vary_output_index in axes(connection_matrix)[2]
        if vary_output_index != fixed_output_index
            distance = distance_matrix[fixed_output_index, vary_output_index]
            overlap = dot(connection_matrix[:, fixed_output_index], connection_matrix[:, vary_output_index])
            points_to_plot = [points_to_plot; [distance overlap]]
        end
    end
    scatter(points_to_plot[:,1], points_to_plot[:,2], legend = false)
end


function plot_receptive_fields_overlap(distance_between_neurons, slope)
    fig = scatter()
    for index in -5:5
        x_value = index*distance_between_neurons
        scatter!((x_value, 1), color = :blue, legend = false, ticks = false)
        plot!([x_value-1/slope, x_value, x_value + 1/slope], [0,1,0], color = :grey)
    end
    return fig
end


# Calculates distance between two floats in [0,1) thought of as points in R/Z
function calculate_circle_distance(phi, theta)
    if phi >=1 || phi <0 || theta >=1 || theta<0
        throw("Invalid inputs to calculate_circle_distance")
    end
    max_val = max(phi, theta)
    min_val = min(phi, theta)
    distance_1 = max_val - min_val
    distance_2 = (1-max_val) + min_val
    circle_distance = min(distance_1, distance_2)
    return circle_distance
end


# Generates neurons uniformly distributed on circle thought of as R/Z
function generate_uniform_neurons_on_circle(num_neurons)
    neuron_positions = collect(LinRange(0,1-1/num_neurons,num_neurons))
    return neuron_positions
end


# Generates a uniform path on R/Z with chosen winding number and number of steps
function generate_circular_path(winding_number, num_steps)
    if winding_number < 0
        throw("Negative winding numer")
    end
    points = collect(LinRange(0, winding_number-winding_number/num_steps, num_steps))
    for index in axes(points)[1]
        points[index] = mod(points[index],1)
    end
    return points
end


# Applies ReLU to each entry of input matrix
function apply_relu_to_matrix(matrix)
    for row_index in axes(matrix)[1]
        for col_index in axes(matrix)[2]
            matrix[row_index, col_index] = max(0, matrix[row_index, col_index])
        end
    end
    return matrix
end


# Defines linear ReLU function with chosen maximum rate and slope
function linear_relu_response_function(max_rate, slope)
    if slope >=0
        throw("Non-negative slope input to linear relu response function")
    else
        function response_function(distance)
            if distance < 0
                throw("Negative distance passed to response_function")
            end
            naive_firing_rate = max_rate + slope*distance
            corrected_firing_rate = max(0, naive_firing_rate)
            return corrected_firing_rate
        end
    end
    return response_function
end


# Input: list of neurons as positions on R/Z, list of points on R/Z representing a path 
# and response_function: distance |-> firing rate
# Output: matrix of dimension (number of nenurons x number of steps on path) whose (i,j) entry is firing rate of neuron i at step j

function calculate_neural_response_matrix(neurons_on_circle, path_on_circle, response_function::Function)
    num_neurons = length(neurons_on_circle)
    num_steps = length(path_on_circle)
    response_matrix = zeros(num_neurons, num_steps)
    for neuron_index in axes(neurons_on_circle)[1]
        for path_index in axes(path_on_circle)[1]
            neuron_position = neurons_on_circle[neuron_index]
            path_position = path_on_circle[path_index]
            distance = calculate_circle_distance(neuron_position, path_position)
            response = response_function(distance)
            response_matrix[neuron_index, path_index] = response
        end
    end
    return response_matrix
end


# Input: list of neurons as positions on R/Z, list of points on R/Z representing a path 
# and response_function: distance |-> firing rate
# Output: matrix of dimension (number of nenurons x number of steps on path) whose (i,j) entry is firing rate of neuron i at step j

function calculate_nonuniform_neural_response_matrix(neurons_on_circle, path_on_circle)
    num_neurons = length(axes(neurons_on_circle)[1])
    num_steps = length(path_on_circle)
    response_matrix = zeros(num_neurons, num_steps)
    for neuron_index in axes(neurons_on_circle)[1]
        neuron_position = neurons_on_circle[neuron_index,1]
        slope = neurons_on_circle[neuron_index,2]
        for path_index in axes(path_on_circle)[1]
            path_position = path_on_circle[path_index]
            distance = calculate_circle_distance(neuron_position, path_position)
            response = linear_relu_response_function(1, slope)(distance)
            response_matrix[neuron_index, path_index] = response
        end
    end
    return response_matrix
end


function generate_random_circle_walk(initial_point, num_steps, step_size_range)
    path = [initial_point]
    counter = 1
    while counter < num_steps
        current_point = path[counter]
        step = rand(Uniform(-step_size_range, step_size_range))
        next_point = current_point + step
        next_point = mod(next_point, 1)
        append!(path, next_point)
        counter = counter + 1
    end
    return path
end



function generate_random_bounded_circle_walk(initial_point, lower_bound, upper_bound, num_steps, step_size_range)
    if initial_point < lower_bound || initial_point > upper_bound
        throw("Unacceptable bounds for 'generate_random_bounded_circle_walk'")
    end
    path = [initial_point]
    counter = 1
    while counter < num_steps
        current_point = path[counter]
        step = rand(Uniform(-step_size_range, step_size_range))
        if current_point + step > upper_bound || current_point + step < lower_bound
            next_point = current_point - step
            append!(path, next_point)
        else
            next_point = current_point + step
            append!(path, next_point)
        end
        counter = counter + 1
    end
    return path
end




function generate_random_skipping_circular_walk(num_walks, num_steps_per_walk, step_size_range)
    full_path = []
    counter = 0
    while counter < num_walks
        start = rand(Uniform(0,1))
        walk = generate_random_circle_walk(start, num_steps_per_walk, step_size_range)
        for position in walk
            append!(full_path, position)
        end
        counter = counter + 1
    end
    return full_path
end


# Given the following
# 1) the response matrix of an initial layer of neurons
# 2) connection matrix between input and output layer of a one-layer network
# 3) fixed bias vector
# This function calculates the neural response matrix of dimension (number output neurons x number time steps)
# of the output layer of neurons
function calculate_output_neural_response_matrix(input_neural_response_matrix, connection_matrix, bias_vector)
    # Check dimensions
    if size(input_neural_response_matrix)[1] != size(connection_matrix)[2] || size(connection_matrix)[1] != length(bias_vector)
        throw("Dimension mismatch in calculate_output_neural_response_matrix")
    end

    
    # Initial multiplication
    output_neural_response_matrix = *(connection_matrix, input_neural_response_matrix)
    
    # Subtract bias vector and apply ReLU
    for row_index in axes(output_neural_response_matrix)[1]
        bias = bias_vector[row_index]
        for col_index in axes(output_neural_response_matrix)[2]
            output_neural_response_matrix[row_index, col_index] = output_neural_response_matrix[row_index, col_index] - bias
            output_neural_response_matrix[row_index, col_index] = max(0.0, output_neural_response_matrix[row_index, col_index])
        end
    end
    return output_neural_response_matrix
end


# Adds a small amount of random noise scaled to each entry
function add_normal_random_noise(matrix, nonzero_entry_percent_stand_dev, zero_entry_stand_dev)
    for row_index in axes(matrix)[1]
        for col_index in axes(matrix)[2]
            entry = matrix[row_index, col_index]
            if entry > 0.0
                distr = Normal(entry, nonzero_entry_percent_stand_dev*entry)
                entry_with_noise = rand(distr)
                entry_with_noise = max(0, entry_with_noise)
                matrix[row_index, col_index] = entry_with_noise
            else
                distr = Normal(0, zero_entry_stand_dev)
                entry_with_noise = rand(distr)
                entry_with_noise = max(0, entry_with_noise)
                matrix[row_index, col_index] = entry_with_noise
            end
        end
    end
    return matrix
end








end