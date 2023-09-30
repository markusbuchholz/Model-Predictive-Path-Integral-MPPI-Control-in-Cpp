// Markus Buchholz 2023
// g++ cpp_mppi.cpp -o t -I/usr/include/eigen3 -I/usr/include/python3.8 -lpython3.8

#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <math.h>
#include <random>

// plot
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// system parameters

float g = 9.81f;
float l = 1.0;   // length of the pole
float m = 0.1;   // mass of the pole
float M = 1.0;   // mass of the cart
float dt = 0.01; // time step
float K = 100;
float T = 200;
float sigma = 1.0;
float lambda = 1.0;

//---------------------------------------------------------------------------------

float gen_rand()
{

    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    return distribution(gen);
}

//---------------------------------------------------------------------------------

Eigen::MatrixXf cart_pole_dynamics(const Eigen::Vector4f &x, float u)
{

    float theta = x(1, 0);
    float theta_dot = x(3, 0);
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    float a = (1.0 / (M + m - m * cos_theta * cos_theta)) * (u + m * sin_theta * (l * theta_dot * theta_dot + g * cos_theta));
    float theta_ddot = (1.0 / (l * (M + m - m * cos_theta * cos_theta))) * (-u * cos_theta - m * l * theta_dot * theta_dot * sin_theta * cos_theta - (M + m) * g * sin_theta);

    Eigen::MatrixXf x_dot(4, 1);  // 4x1 matrix
    Eigen::MatrixXf x_next(4, 1); // 4x1 matrix
    x_dot << x(2, 0), x(3, 0), a, theta_ddot;
    x_next = x + x_dot * dt;

    return x_next;
}
//---------------------------------------------------------------------------------

float cost_function(const Eigen::Vector4f &x, float u)
{

    // Define the diagonal matrix Q
    Eigen::Vector4f q_values(1.0f, 10.0f, 1.0f, 10.0f);
    Eigen::DiagonalMatrix<float, 4> Q(q_values);

    // Define the matrix R
    Eigen::Matrix<float, 1, 1> R;
    R << 0.1f;

    // Compute the cost
    float cost = x.transpose() * Q * x + u * R(0, 0) * u;

    return cost;
}

//---------------------------------------------------------------------------------

float mppi(const Eigen::Vector4f &x)
{
    const float dt = 0.01f;
    int T = 10;
    int K = 100;

    Eigen::MatrixXf U = Eigen::MatrixXf::Random(K, T) * sigma;
    std::vector<Eigen::MatrixXf> X(K, Eigen::MatrixXf::Zero(T + 1, 4));
    for (int k = 0; k < K; ++k)
    {
        X[k].row(0) = x.transpose();
    }

    Eigen::VectorXf costs = Eigen::VectorXf::Zero(K);

    for (int k = 0; k < K; ++k)
    {
        for (int t = 0; t < T; ++t)
        {
            X[k].row(t + 1) = cart_pole_dynamics(X[k].row(t), U(k, t)).transpose();
            costs(k) += cost_function(X[k].row(t + 1), U(k, t));
        }
    }

    Eigen::VectorXf weights = (-lambda * (costs.array() - costs.minCoeff())).exp();
    weights /= weights.sum();

    Eigen::VectorXf u_star = (weights.asDiagonal() * U).colwise().sum();

    return u_star(0);
}

//---------------------------------------------------------------------------------

void plot(std::vector<float> time, std::vector<float> cart_pos, std::vector<float> pole_angle, std::vector<float> cart_velo, std::vector<float> pole_velo)
{

    plt::title("Model Predictive Path Integral (MPPI) Control for cart pole");
    plt::named_plot("cart_pos", time, cart_pos);
    plt::named_plot("pole_angle", time, pole_angle);
    plt::named_plot("cart_velo", time, cart_velo);
    plt::named_plot("pole_velo", time, pole_velo);
    plt::xlabel("time");
    plt::ylabel("Y");
    plt::legend();

    plt::show();
}

//---------------------------------------------------------------------------------

int main()
{

    int Tx = 200;
    Eigen::Vector4f x(0.0f, 0.1f, 0.0f, 0.0f); // initial state [x, theta, x_dot, theta_dot]
    Eigen::Matrix<float, Eigen::Dynamic, 4> X = Eigen::Matrix<float, Eigen::Dynamic, 4>::Zero(T, 4);
    Eigen::VectorXf U = Eigen::VectorXf::Zero(T);

    std::vector<float> time;
    std::vector<float> cart_pos;
    std::vector<float> pole_angle;
    std::vector<float> cart_velo;
    std::vector<float> pole_velo;

    for (int t = 0; t < Tx - 1; ++t)
    {
        U(t) = mppi(x);
        x = cart_pole_dynamics(x, U(t));
        X.row(t + 1) = x.transpose();
        time.push_back(t);
        cart_pos.push_back(X.row(t + 1)(0));
        pole_angle.push_back(X.row(t + 1)(1));
        cart_velo.push_back(X.row(t + 1)(2));
        pole_velo.push_back(X.row(t + 1)(3));
    }
    plot(time, cart_pos, pole_angle, cart_velo, pole_velo);

    return 0;
}
