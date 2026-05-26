#ifndef DYNAMIC_MODEL_CAR_HPP
#define DYNAMIC_MODEL_CAR_HPP

#include "lion/math/matrix_extensions.h"
#include "lion/math/euler_angles.h"

template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
inline auto Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::transform_states_to_inputs(const std::array<Timeseries_t, number_of_states>& states, 
    const std::array<Timeseries_t, number_of_controls>& controls ) const -> std::array<Timeseries_t,number_of_inputs>
{
    static_assert(number_of_states == number_of_inputs);
    // (1) Define inputs
    auto inputs = states;

    // (2) Duplicate the chassis: we do not want to modify its internal state
    auto chassis = get_chassis();
    chassis.transform_states_to_inputs(states, controls, inputs);

    return inputs;
}

template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
inline auto Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::ode
    (const std::array<Timeseries_t,number_of_states>& states, const std::array<Timeseries_t,number_of_controls>& controls, scalar time) -> std::array<Timeseries_t,number_of_states>
{
	static_assert(number_of_states == number_of_inputs);
	
    const auto inputs = transform_states_to_inputs(states, controls);
    return (*this)(inputs,controls,time).dstates_dt;
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
template<typename T>
void Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::set_parameter(const std::string& parameter, const T value) 
{ 
    // (1) Set the value of the parameter
    get_chassis().set_parameter(parameter,value); 

    // (2) Check if the parameter is an optimization parameter, if so, change its value also there.
    if constexpr ( std::is_same_v<T,scalar> || std::is_same_v<T,CppAD::AD<scalar>> )
    {
        const auto it_p = std::find_if(base_type::get_parameters().begin(), base_type::get_parameters().end(), [&](const auto& p) -> auto { return p.get_path() == parameter; });

        if ( it_p != base_type::get_parameters().cend() )
            std::fill(it_p->get_values().begin(), it_p->get_values().end(), value);
    }
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
void Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::set_wet_surface(
    scalar base_grip_multiplier,
    scalar dry_line_penalty,
    scalar dry_line_width,
    const std::vector<scalar>& arclength,
    const std::vector<scalar>& dry_line_lateral_displacement)
{
    if ( arclength.size() != dry_line_lateral_displacement.size() )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> arclength and dry line vectors must have the same size");

    if ( arclength.size() < 2 )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> at least two dry line points are required");

    if ( !std::is_sorted(arclength.cbegin(), arclength.cend()) )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> arclength must be sorted");

    if ( dry_line_width <= 0.0 )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> dry line width must be positive");

    if ( base_grip_multiplier <= 0.0 )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> base grip multiplier must be positive");

    if ( dry_line_penalty < 0.0 || dry_line_penalty >= 1.0 )
        throw fastest_lap_exception("[ERROR] Dynamic_model_car::set_wet_surface -> dry line penalty must be in [0,1)");

    _wet_surface_enabled = true;
    _wet_base_grip_multiplier = base_grip_multiplier;
    _wet_dry_line_penalty = dry_line_penalty;
    _wet_dry_line_width = dry_line_width;
    _wet_surface_arclength = arclength;
    _wet_surface_dry_line_lateral_displacement = dry_line_lateral_displacement;
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
void Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::clear_wet_surface()
{
    _wet_surface_enabled = false;
    _wet_surface_arclength.clear();
    _wet_surface_dry_line_lateral_displacement.clear();
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
scalar Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::interpolate_dry_line_lateral_displacement(scalar s) const
{
    auto it = std::upper_bound(_wet_surface_arclength.cbegin(), _wet_surface_arclength.cend(), s);

    if ( it == _wet_surface_arclength.cbegin() )
        return _wet_surface_dry_line_lateral_displacement.front();

    if ( it == _wet_surface_arclength.cend() )
        return _wet_surface_dry_line_lateral_displacement.back();

    const auto index_right = static_cast<size_t>(std::distance(_wet_surface_arclength.cbegin(), it));
    const auto index_left = index_right - 1;
    const scalar s_left = _wet_surface_arclength[index_left];
    const scalar s_right = _wet_surface_arclength[index_right];
    const scalar xi = (s - s_left)/(s_right - s_left);

    return _wet_surface_dry_line_lateral_displacement[index_left]
        + xi*(_wet_surface_dry_line_lateral_displacement[index_right]
              - _wet_surface_dry_line_lateral_displacement[index_left]);
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
void Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::apply_wet_surface_grip_multiplier(
    const std::array<Timeseries_t,number_of_inputs>& inputs,
    scalar s)
{
    Timeseries_t grip_multiplier = 1.0;

    if constexpr (road_is_curvilinear<RoadModel_t>::value)
    {
        if ( _wet_surface_enabled )
        {
            const scalar dry_line_n = interpolate_dry_line_lateral_displacement(s);
            const Timeseries_t dn = inputs[RoadModel_t::input_names::lateral_displacement] - dry_line_n;
            const scalar inv_width = 1.0/_wet_dry_line_width;
            const Timeseries_t dry_line_penalty = _wet_dry_line_penalty*exp(-0.5*dn*dn*inv_width*inv_width);

            grip_multiplier = _wet_base_grip_multiplier*(1.0 - dry_line_penalty);
        }
    }

    _chassis.get_front_axle().template get_tire<0>().set_grip_multiplier(grip_multiplier);
    _chassis.get_front_axle().template get_tire<1>().set_grip_multiplier(grip_multiplier);
    _chassis.get_rear_axle().template get_tire<0>().set_grip_multiplier(grip_multiplier);
    _chassis.get_rear_axle().template get_tire<1>().set_grip_multiplier(grip_multiplier);
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
inline auto Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::operator()
    (const std::array<Timeseries_t,number_of_inputs>& inputs, const std::array<Timeseries_t,number_of_controls>& controls, scalar time) -> Dynamics_equations
{
    // (1) Initialize outputs
    Dynamics_equations dynamics_equations;

    auto& states              = dynamics_equations.states;
    auto& dstates_dt          = dynamics_equations.dstates_dt;

    // (2) Set the variable parameters
    for (auto const& parameter : base_type::get_parameters() )
        get_chassis().set_parameter(parameter.get_path(), parameter(time));

    // (3) Set state and controls
    _chassis.set_state_and_controls(inputs,controls);
    _road.set_state_and_controls(time,inputs,controls);
    apply_wet_surface_grip_multiplier(inputs,time);

    // (4) Update

    // (4.1) Update road frenet frame: velocities are borrowed from the car
    _road.update(_chassis.get_u(), _chassis.get_v(), _chassis.get_yaw_rate_radps());

    // (4.2) Update the car dynamic model: position are borrowed from the road
    const Vector3d<Timeseries_t>     ground_position_vector_m = std::as_const(_road).get_position();
    Euler_angles<scalar>             track_euler_angles_rad;
    Euler_angles<Timeseries_t>       track_euler_angles_dot_radps;
    Timeseries_t                     track_heading_angle_rad;
    Timeseries_t                     track_heading_angle_dot_radps;
    const Timeseries_t&              ground_velocity_z_mps = std::as_const(_road).get_ground_vertical_velocity();
    
    if constexpr (road_is_curvilinear<decltype(_road)>::value)
    {
        track_euler_angles_rad         = _road.get_euler_angles();
        track_euler_angles_dot_radps   = _road.get_euler_angles_dot();
        track_heading_angle_rad        = _road.get_track_heading_angle();
        track_heading_angle_dot_radps  = _road.get_track_heading_angle_dot();
    }
    else
    {
        track_euler_angles_rad        = { 0.0, 0.0, 0.0 };
        track_euler_angles_dot_radps  = { 0.0, 0.0, 0.0 };
        track_heading_angle_rad       = std::as_const(_road).get_psi();
        track_heading_angle_dot_radps = _chassis.get_yaw_rate_radps();
    }

    _chassis.update(ground_position_vector_m, track_euler_angles_rad, track_heading_angle_rad, track_euler_angles_dot_radps, track_heading_angle_dot_radps, ground_velocity_z_mps);

    // (5) Get state and state time derivative
    _chassis.get_state_and_state_derivative(states,dstates_dt);
    _road.get_state_and_state_derivative(states,dstates_dt);

    // (6) Scale the temporal parameter to curvilinear if needed
    for (auto it = dstates_dt.begin(); it != dstates_dt.end(); ++it)
        (*it) *= std::as_const(_road).get_dtimeds();

    return dynamics_equations;
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
auto Dynamic_model_car<Timeseries_t, Chassis_t, RoadModel_t>::equations
(const std::array<scalar, number_of_inputs>& inputs, const std::array<scalar, number_of_controls>& controls, scalar time) -> Equations
{
    // (1) Put the states into a single vector, which will be declared as independent variables
    constexpr const size_t n_total = number_of_inputs + number_of_controls;
    std::vector<CppAD::AD<double>> inputs_all_ad(n_total);

    std::copy(inputs.cbegin(), inputs.cend(), inputs_all_ad.begin());
    std::copy(controls.cbegin(), controls.cend(), inputs_all_ad.begin() + number_of_inputs);

    // (2) Declare the contents of x0 as the independent variables
    CppAD::Independent(inputs_all_ad);

    // (3) Create new inputs to the operator() of the vehicle 
    std::array<CppAD::AD<double>, number_of_inputs>   inputs_ad;
    std::array<CppAD::AD<double>, number_of_controls> controls_ad;

    std::copy_n(inputs_all_ad.cbegin(), number_of_inputs, inputs_ad.begin());
    std::copy(inputs_all_ad.cbegin() + number_of_inputs, inputs_all_ad.cend(), controls_ad.begin());

    // (4) Call operator(), transform arrays to vectors
    auto dynamic_equations = (*this)(inputs_ad, controls_ad, 0.0);
    const auto& state_ad = dynamic_equations.states;
    const auto& dstates_dt_ad = dynamic_equations.dstates_dt;

    // (5) Concatenate [states, dstates_dt]
    std::vector<CppAD::AD<double>> outputs_all_ad(state_ad.cbegin(), state_ad.cend());
    outputs_all_ad.insert(outputs_all_ad.end(), dstates_dt_ad.cbegin(), dstates_dt_ad.cend());

    // (6) Create the AD functions and stop the recording
    CppAD::ADFun<double> f;
    f.Dependent(inputs_all_ad, outputs_all_ad);

    // (7) Transform inputs to double, to evaluate the functions
    std::vector<scalar> inputs_all(inputs_all_ad.size());
    std::transform(inputs_all_ad.cbegin(), inputs_all_ad.cend(), inputs_all.begin(),
        [](const auto& ad) -> auto { return Value(ad); });
    
    // (8) Evaluate y = f(q0,u0,0)
    auto outputs_all = f.Forward(0, inputs_all);
    auto jacobian_outputs_all = f.Jacobian(inputs_all);
    
    std::vector<std::vector<double>> hessian_outputs_all(2*number_of_states);

    for (size_t i_output = 0; i_output < 2*number_of_states; ++i_output)
        hessian_outputs_all[i_output] = f.Hessian(inputs_all,i_output);

    // (9) Fill the solution struct
    Equations solution;

    // (9.1) Solution
    std::copy_n(outputs_all.cbegin()                       , number_of_states, solution.states.begin());
    std::copy_n(outputs_all.cbegin() + number_of_states    , number_of_states, solution.dstates_dt.begin());

    // (9.2) Jacobians
    for (size_t i = 0; i < number_of_states; ++i)
        for (size_t j = 0; j < n_total; ++j)
            // The CppAD jacobian is sorted row major, [dy1/dx1, ..., dy1/dxn, dy2/dx1, ..., dy2/dxn, ...]
            solution.jacobian_states[i][j] = jacobian_outputs_all[j + n_total*i];

    for (size_t i = 0; i < number_of_states; ++i)
        for (size_t j = 0; j < n_total; ++j)
            // The CppAD jacobian is sorted row major, [dy1/dx1, ..., dy1/dxn, dy2/dx1, ..., dy2/dxn, ...]
            solution.jacobian_dstates_dt[i][j] = jacobian_outputs_all[j + n_total*(i+number_of_states)];

    // (9.3) Hessians
    for (size_t var = 0; var < number_of_states; ++var)
        for (size_t i = 0; i < n_total; ++i)
            for (size_t j = 0; j < n_total; ++j)
            {
                solution.hessian_states[var][i][j] = hessian_outputs_all[var][j + n_total*i];
    
                // Check its symmetry
                assert(std::abs(hessian_outputs_all[var][j + n_total*i]- hessian_outputs_all[var][i + n_total*j])
                         < 1.0e-10*std::max(1.0,std::abs(hessian_outputs_all[var][j+n_total*i])));
            }

    for (size_t var = 0; var < number_of_states; ++var)
        for (size_t i = 0; i < n_total; ++i)
            for (size_t j = 0; j < n_total; ++j)
            {
                solution.hessian_dstates_dt[var][i][j] = hessian_outputs_all[var+number_of_states][j + n_total*i];
    
                // Check its symmetry
                assert(std::abs(hessian_outputs_all[var+number_of_states][j + n_total*i]- hessian_outputs_all[var+number_of_states][i + n_total*j])
                         < 1.0e-10*std::max(1.0,std::abs(hessian_outputs_all[var+number_of_states][j+n_total*i])));
            }


    return solution;
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
inline auto Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::get_state_and_control_names() const -> std::tuple<std::string,std::array<std::string,number_of_inputs>,std::array<std::string,number_of_controls>>
{
    std::string key_name;
    std::array<std::string,number_of_inputs> inputs_names;
    std::array<std::string,number_of_controls> controls_names;

    RoadModel_t::set_state_and_control_names(key_name,inputs_names,controls_names);
    _chassis.set_state_and_control_names(inputs_names,controls_names);

    return {key_name,inputs_names,controls_names};
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
inline auto Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::get_state_and_control_upper_lower_and_default_values() const -> State_and_control_upper_lower_and_default_values
{
    // (1) Define outputs
    State_and_control_upper_lower_and_default_values values_all;

    auto& inputs_def = values_all.inputs_def;
    auto& inputs_lb  = values_all.inputs_lb;
    auto& inputs_ub  = values_all.inputs_ub;

    auto& controls_def = values_all.controls_def;
    auto& controls_lb = values_all.controls_lb;
    auto& controls_ub = values_all.controls_ub;

    // (2) Outputs are filled by chassis
    _chassis.set_state_and_control_upper_lower_and_default_values(inputs_def, inputs_lb, inputs_ub, 
        controls_def, controls_lb, controls_ub);

    // (3) Outputs are filled by road
    _road.set_state_and_control_upper_lower_and_default_values(inputs_def, inputs_lb, inputs_ub,
            controls_def, controls_lb, controls_ub);

    // (4) Return
    return values_all;
}


template<typename Timeseries_t, typename Chassis_t, typename RoadModel_t>
bool Dynamic_model_car<Timeseries_t,Chassis_t,RoadModel_t>::is_ready() const
{
    return _chassis.is_ready();
}

#endif
