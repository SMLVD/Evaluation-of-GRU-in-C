#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <limits>

std::vector<double> print_vector(const std::vector<double>& _input_vector)
{
	for (size_t i = 0; i < _input_vector.size(); i++)
	{
		std::cout << _input_vector[i] << std::endl;
	}
	return _input_vector;
}

std::vector<std::vector<double>> print_matrix(const std::vector<std::vector<double>>& _input_matrix)
{
	size_t _input_row = _input_matrix.size();
	size_t _input_column = _input_matrix[0].size();
	for (size_t i = 0; i < _input_row; i++)
	{
		for (size_t j = 0; j < _input_column; j++)
		{
			std::cout << _input_matrix[i][j] << " ";
		}
		std::cout << "" << std::endl;
	}
	return _input_matrix;
}

std::vector<double> sigmoid(const std::vector<double>& _input_vector)
{
	std::vector<double> _return_vector(_input_vector.size());
	for (size_t i = 0; i < _input_vector.size(); i++)
	{
		_return_vector[i] = 1 / (1 + std::exp(-_input_vector[i]));
	}
	return _return_vector;
}

double dot_multi(const std::vector<double>& _input_a, const std::vector<double>& _input_b) 
{
	double _return_sum = 0.0;
	for (size_t i = 0; i < _input_a.size(); i++)
	{
		_return_sum = _return_sum + _input_a[i] * _input_b[i];
	}
	return _return_sum;
}

std::vector<std::vector<double>> trans_m(const std::vector<std::vector<double>>& _input_matrix)
{
	size_t _input_row = _input_matrix.size();
	size_t _input_column = _input_matrix[0].size();
	std::vector<std::vector<double>> _return_matrix(_input_column, std::vector<double>(_input_row));
	for (size_t i = 0; i < _input_row; i++)
	{
		for (size_t j = 0; j < _input_column; j++)
		{
			_return_matrix[j][i] = _input_matrix[i][j];
		}
	}
	return _return_matrix;
}

std::vector<std::vector<double>> trans_v(const std::vector<double>& _input_vector)
{
	std::vector<std::vector<double>> _return_matrix(_input_vector.size(), std::vector<double>(1));
	for (size_t i = 0; i < _input_vector.size(); i++)
	{
		_return_matrix[i][0] = _input_vector[i];
	}
	return _return_matrix;
}

std::vector<double> forward_tilde_h_t(const std::vector<std::vector<double>>& _U_h, const std::vector<double>& _r_t, const std::vector<double>& _h_t_1, const std::vector<std::vector<double>>& _W_h, const std::vector<double>& _s_t, const std::vector<double>& _b_h)
{
	std::vector<double> _calc_2(_b_h.size());
	std::vector<double> _return_double(_b_h.size());
	for (size_t i = 0; i < _b_h.size(); i++)
	{
		double _calc_1 = 0.0;
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1 = _calc_1 + _W_h[i][j] * _s_t[j];
		}
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1 = _calc_1 + _U_h[i][j] * (_r_t[j] * _h_t_1[j]);
		}
		_calc_2[i] = std::tanh(_calc_1 + _b_h[i]);
	}
	return _calc_2;
}

std::vector<double> forward_h_t(const std::vector<double>& _z_t, const std::vector<double>& _h_t_1, const std::vector<double>& _tilde_h_t)
{
	std::vector<double> _calc_1(_z_t.size());
	std::vector<double> _return_double(_z_t.size());
	for (size_t i = 0; i < _z_t.size(); i++)
	{
		_calc_1[i] = (1 - _z_t[i]) * _tilde_h_t[i] + _h_t_1[i] * _z_t[i];
	}
	_return_double = _calc_1;
	return _return_double;
}

std::vector<double> forward_z_t(const std::vector<std::vector<double>>& _U_z, const std::vector<double>& _h_t_1, const std::vector<std::vector<double>>& _W_z, const std::vector<double>& _s_t, const std::vector<double>& _b_z)
{
	std::vector<double> _calc_2(_b_z.size());
	std::vector<double> _return_vector(_b_z.size());
	for (size_t i = 0; i < _b_z.size(); i++)
	{
		double _calc_1 = 0.0;
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1 = _calc_1 + _U_z[i][j] * _h_t_1[j];
		}
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1 = _calc_1 + _W_z[i][j] * _s_t[j];
		}
		_calc_2[i] = _calc_1 + _b_z[i];
	}
	_return_vector = sigmoid(_calc_2);
	return _return_vector;
}

std::vector<double> forward_r_t(const std::vector<std::vector<double>>& _U_r, const std::vector<double>& _h_t_1, const std::vector<std::vector<double>>& _W_r, const std::vector<double>& _s_t, const std::vector<double>& _b_r)
{
	std::vector<double> _calc_2(_b_r.size());
	std::vector<double> _return_vector(_b_r.size());
	for (size_t i = 0; i < _b_r.size(); i++)
	{
		double _calc_1 = 0.0;
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1 = _calc_1 + _U_r[i][j] * _h_t_1[j];
		}
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1 = _calc_1 + _W_r[i][j] * _s_t[j];
		}
		_calc_2[i] = _calc_1 + _b_r[i];
	}
	_return_vector = sigmoid(_calc_2);
	return _return_vector;
}

std::vector<double> error(const std::vector<double>& _prediction, const std::vector<double>& _target)
{
	std::vector<double> _return_vector(_prediction.size());
	for (size_t i = 0; i < _prediction.size(); i++)
	{
		_return_vector[i] = _prediction[i] - _target[i];
	}
	return _return_vector;
}

std::vector<double> calc_mse(const std::vector<double>& _prediction, const std::vector<double>& _target)
{
	std::vector<double> _return_vector(_prediction.size());
	for (size_t i = 0; i < _prediction.size(); i++)
	{
		_return_vector[i] = (_target[i] - _prediction[i]) * (_target[i] - _prediction[i]);
	}
	return _return_vector;
}

std::vector<double> backward_p_z_t(const std::vector<double>& _p_h_t, const std::vector<double>& _tilde_h_t, const std::vector<double>& _h_t_1)
{
	std::vector<double> _calc_1(_tilde_h_t.size());
	std::vector<double> _return_vector(_tilde_h_t.size());
	for (size_t i = 0; i < _tilde_h_t.size(); i++)
	{
		_calc_1[i] = _p_h_t[i] * (_h_t_1[i] - _tilde_h_t[i]);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_tilde_h_t(const std::vector<double>& _p_h_t, const std::vector<double>& _z_t)
{
	std::vector<double> _calc_1(_p_h_t.size());
	std::vector<double> _return_vector(_p_h_t.size());
	for (size_t i = 0; i < _p_h_t.size(); i++)
	{
		_calc_1[i] = _p_h_t[i] * (1 - _z_t[i]);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_h_t_1_FHState(const std::vector<double>& _p_h_t, const std::vector<double>& _z_t)
{
	std::vector<double> _calc_1(_p_h_t.size());
	std::vector<double> _return_vector(_p_h_t.size());
	for (size_t i = 0; i < _p_h_t.size(); i++)
	{
		_calc_1[i] = _p_h_t[i] * _z_t[i];
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_a_t(const std::vector<double>& _p_tilde_h_t, const std::vector<double>& _tilde_h_t)
{
	std::vector<double> _calc_1(_tilde_h_t.size());
	std::vector<double> _return_vector(_tilde_h_t.size());
	for (size_t i = 0; i < _tilde_h_t.size(); i++)
	{
		_calc_1[i] = _p_tilde_h_t[i] * (1 - (_tilde_h_t[i] * _tilde_h_t[i]));
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_r_t(const std::vector<std::vector<double>>& _U_h, const std::vector<double>& _h_t_1, const std::vector<double>& _p_a_t)
{
	std::vector<double> _calc_1(_h_t_1.size());
	std::vector<std::vector<double>> _U_h_trans = trans_m(_U_h);
	std::vector<double> _return_vector(_h_t_1.size());
	for (size_t i = 0; i < _h_t_1.size(); i++)
	{
		_calc_1[i] = dot_multi(_U_h_trans[i], _p_a_t) * _h_t_1[i];
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_h_t_1_CHState(const std::vector<std::vector<double>>& _U_h, const std::vector<double>& _r_t, const std::vector<double>& _p_a_t)
{
	std::vector<double> _calc_1(_r_t.size());
	std::vector<std::vector<double>> _U_h_trans = trans_m(_U_h);
	std::vector<double> _return_vector(_r_t.size());
	for (size_t i = 0; i < _r_t.size(); i++)
	{
		_calc_1[i] = dot_multi(_U_h_trans[i], _p_a_t) * _r_t[i];
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_b_r(const std::vector<double>& _p_r_t, const std::vector<double>& _r_t)
{
	std::vector<double> _calc_1(_r_t.size());
	std::vector<double> _return_vector(_r_t.size());
	for (size_t i = 0; i < _r_t.size(); i++)
	{
		_calc_1[i] = _p_r_t[i] * _r_t[i] * (1 - _r_t[i]);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_h_t_1_RState(const std::vector<std::vector<double>>& _U_r, const std::vector<double>& _p_b_r)
{
	std::vector<double> _calc_1(_p_b_r.size());
	std::vector<std::vector<double>> _U_r_trans = trans_m(_U_r);
	std::vector<double> _return_vector(_p_b_r.size());
	for (size_t i = 0; i < _p_b_r.size(); i++)
	{
		_calc_1[i] = dot_multi(_U_r_trans[i], _p_b_r);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_b_z(const std::vector<double>& _p_z_t, const std::vector<double>& _z_t)
{
	std::vector<double> _calc_1(_z_t.size());
	std::vector<double> _return_vector(_z_t.size());
	for (size_t i = 0; i < _z_t.size(); i++)
	{
		_calc_1[i] = _p_z_t[i] * _z_t[i] * (1 - _z_t[i]);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<double> backward_p_h_t_1_UState(const std::vector<std::vector<double>>& _U_z, const std::vector<double>& _p_b_z)
{
	std::vector<double> _calc_1(_p_b_z.size());
	std::vector<std::vector<double>> _U_z_trans = trans_m(_U_z);
	std::vector<double> _return_vector(_p_b_z.size());
	for (size_t i = 0; i < _p_b_z.size(); i++)
	{
		_calc_1[i] = dot_multi(_U_z_trans[i], _p_b_z);
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<std::vector<double>> backward_PG_W_r(const std::vector<double>& _p_b_r, const std::vector<double>& _s_t)
{
	std::vector<std::vector<double>> _calc_1(_p_b_r.size(), std::vector<double>(_s_t.size()));
	std::vector<std::vector<double>> _return_matrix(_p_b_r.size(), std::vector<double>(_s_t.size()));
	for (size_t i = 0; i < _p_b_r.size(); i++)
	{
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1[i][j] = _p_b_r[i] * _s_t[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_U_r(const std::vector<double>& _p_b_r, const std::vector<double>& _h_t_1)
{
	std::vector<std::vector<double>> _calc_1(_p_b_r.size(), std::vector<double>(_h_t_1.size()));
	std::vector<std::vector<double>> _return_matrix(_p_b_r.size(), std::vector<double>(_h_t_1.size()));
	for (size_t i = 0; i < _p_b_r.size(); i++)
	{
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1[i][j] = _p_b_r[i] * _h_t_1[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_W_z(const std::vector<double>& _p_b_z, const std::vector<double>& _s_t)
{
	std::vector<std::vector<double>> _calc_1(_p_b_z.size(), std::vector<double>(_s_t.size()));
	std::vector<std::vector<double>> _return_matrix(_p_b_z.size(), std::vector<double>(_s_t.size()));
	for (size_t i = 0; i < _p_b_z.size(); i++)
	{
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1[i][j] = _p_b_z[i] * _s_t[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_U_z(const std::vector<double>& _p_b_z, const std::vector<double>& _h_t_1)
{
	std::vector<std::vector<double>> _calc_1(_p_b_z.size(), std::vector<double>(_h_t_1.size()));
	std::vector<std::vector<double>> _return_matrix(_p_b_z.size(), std::vector<double>(_h_t_1.size()));
	for (size_t i = 0; i < _p_b_z.size(); i++)
	{
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1[i][j] = _p_b_z[i] * _h_t_1[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_W_h(const std::vector<double>& _p_a_t, const std::vector<double>& _s_t)
{
	std::vector<std::vector<double>> _calc_1(_p_a_t.size(), std::vector<double>(_s_t.size()));
	std::vector<std::vector<double>> _return_matrix(_p_a_t.size(), std::vector<double>(_s_t.size()));
	for (size_t i = 0; i < _p_a_t.size(); i++)
	{
		for (size_t j = 0; j < _s_t.size(); j++)
		{
			_calc_1[i][j] = _p_a_t[i] * _s_t[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_U_h(const std::vector<double>& _p_a_t, const std::vector<double>& _h_t_1, const std::vector<double>& _r_t)
{
	std::vector<std::vector<double>> _calc_1(_h_t_1.size(), std::vector<double>(_h_t_1.size()));
	std::vector<std::vector<double>> _return_matrix(_h_t_1.size(), std::vector<double>(_h_t_1.size()));
	for (size_t i = 0; i < _h_t_1.size(); i++)
	{
		for (size_t j = 0; j < _h_t_1.size(); j++)
		{
			_calc_1[i][j] = _p_a_t[i] * (_r_t[j] * _h_t_1[j]);
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<std::vector<double>> backward_PG_W_y(const std::vector<double>& _prediction, const std::vector<double>& _h_t)
{
	std::vector<std::vector<double>> _calc_1(_prediction.size(), std::vector<double>(_h_t.size()));
	std::vector<std::vector<double>> _return_matrix(_prediction.size(), std::vector<double>(_h_t.size()));
	for (size_t i = 0; i < _prediction.size(); i++)
	{
		for (size_t j = 0; j < _h_t.size(); j++)
		{
			_calc_1[i][j] = _prediction[i] * _h_t[j];
		}
	}
	_return_matrix = _calc_1;
	return _return_matrix;
}

std::vector<double> middle_PG_W_y(const std::vector<std::vector<double>>& _W_y, const std::vector<double>& _error)
{
	std::vector<double> _return_vector(_W_y[0].size());
	for (size_t i = 0; i < _W_y[0].size(); i++)
	{
		for (size_t j = 0; j < _error.size(); j++)
		{
			_return_vector[i] = _return_vector[i] + _error[j] * _W_y[j][i];
		}
	}
	return _return_vector;
}

std::vector<double> sum_gradient(const std::vector<double>& _a_p_h_t_1, const std::vector<double>& _b_p_h_t_1, const std::vector<double>& _c_p_h_t_1, const std::vector<double>& _d_p_h_t_1)
{
	std::vector<double> _calc_1(_a_p_h_t_1.size());
	std::vector<double> _return_vector(_a_p_h_t_1.size());
	for (size_t i = 0; i < _a_p_h_t_1.size(); i++)
	{
		_calc_1[i] = _a_p_h_t_1[i] + _b_p_h_t_1[i] + _c_p_h_t_1[i] + _d_p_h_t_1[i];
	}
	_return_vector = _calc_1;
	return _return_vector;
}

std::vector<std::vector<double>> update_matrix(const std::vector<std::vector<double>>& _input_matrix_a, const std::vector<std::vector<double>>& _input_matrix_b, const double& _rating)
{
	size_t _input_row = _input_matrix_a.size();
	size_t _input_column = _input_matrix_a[0].size();
	std::vector<std::vector<double>> _return_matrix(_input_row, std::vector<double>(_input_column));
	for (size_t i = 0; i < _input_row; i++)
	{
		for (size_t j = 0; j < _input_column; j++)
		{
			_return_matrix[i][j] = _input_matrix_a[i][j] - _rating * _input_matrix_b[i][j];
		}
	}
	return _return_matrix;
}

std::vector<double> update_vector(const std::vector<double>& _input_vector_a, const std::vector<double>& _input_vector_b, const double& _rating)
{
	std::vector<double> _return_vector(_input_vector_a.size());
	for (size_t i = 0; i < _input_vector_a.size(); i++)
	{
		_return_vector[i] = _input_vector_a[i] - _rating * _input_vector_b[i];
	}
	return _return_vector;
}

std::vector<double> update_vector_MSE(const std::vector<double>& _input_vector_a, const std::vector<double>& _input_vector_b)
{
	std::vector<double> _return_vector(_input_vector_a.size());
	for (size_t i = 0; i < _input_vector_a.size(); i++)
	{
		_return_vector[i] = _input_vector_a[i] + _input_vector_b[i];
	}
	return _return_vector;
}

double update_double(const double& _input_double_a, const double& _input_double_b, const double& _rating)
{
	double _return_double = 0.0;
	_return_double = _input_double_a - _input_double_b * _rating;
	return _return_double;
}

std::vector<double> prediction(const std::vector<std::vector<double>>& _W_y, const std::vector<double>& _h_t, const std::vector<double>& _b_y)
{
	std::vector<double> _return_double(_W_y.size());
	std::vector<double> _calc1(_W_y.size());
	for (size_t i = 0; i < _W_y.size(); i++)
	{
		for (size_t j = 0; j < _W_y[0].size(); j++)
		{
			_calc1[i] = _calc1[i] + _W_y[i][j] * _h_t[j];
		}
		_return_double[i] = _calc1[i] + _b_y[i];
	}
	return _return_double;
}

std::vector<std::vector<double>> create_matrix(const size_t& _input_row, const size_t& _input_column)
{
	std::vector<std::vector<double>> _return_matrix(_input_row, std::vector<double>(_input_column));
	for (size_t i = 0; i < _input_row; i++)
	{
		for (size_t j = 0; j < _input_column; j++)
		{
			_return_matrix[i][j] = ((double)rand() / RAND_MAX * 0.2 - 0.10);
		}
	}
	return _return_matrix;
}

std::vector<double> create_vector(const size_t& _size)
{
	std::vector<double> _return_vector(_size);
	for (size_t i = 0; i < _size; i++)
	{
		_return_vector[i] = ((double)rand() / RAND_MAX * 0.2 - 0.1);
	}
	return _return_vector;
}

std::vector<double> create_zero_vector(const size_t& _size)
{
	std::vector<double> _return_vector(_size);
	for (size_t i = 0; i < _size; i++)
	{
		_return_vector[i] = 0.00;
	}
	return _return_vector;
}

std::vector<double> create_val_vector(const size_t& _size, const double& _val)
{
	std::vector<double> _return_vector(_size);
	for (size_t i = 0; i < _size; i++)
	{
		_return_vector[i] = _val;
	}
	return _return_vector;
}

struct Forward_Parameter
{
	std::vector<double> z_t;
	std::vector<double> r_t;
	std::vector<double> tilde_h_t;
	std::vector<double> h_t;
	std::vector<double> pred;
	std::vector<double> error;
	std::vector<double> mse;
	std::vector<double> p_h_t;
	std::vector<double> h_t_1;
};

std::vector<Forward_Parameter> forward_pass(const std::vector<double>& _h_t_1, const std::vector<std::vector<double>>& _target, const std::vector<double>& _b_y, const std::vector<std::vector<double>>& _W_y, const size_t& _timesteps, std::vector<std::vector<double>>& _U_r, std::vector<std::vector<double>>& _U_z, std::vector<std::vector<double>>& _U_h, std::vector<std::vector<double>>& _W_r, std::vector<std::vector<double>>& _W_z, std::vector<std::vector<double>>& _W_h, std::vector<double>& _b_r, std::vector<double>& _b_z, std::vector<double>& _b_h, std::vector<std::vector<double>>& _s_t)
{
	std::vector<Forward_Parameter> _return(_timesteps);
	_return[0].h_t_1 = _h_t_1;
	for (size_t i = 0; i < _timesteps; i++)
	{
		_return[i].z_t = forward_z_t(_U_z, _return[i].h_t_1, _W_z, _s_t[i], _b_z);
		_return[i].r_t = forward_r_t(_U_r, _return[i].h_t_1, _W_r, _s_t[i], _b_r);
		_return[i].tilde_h_t = forward_tilde_h_t(_U_h, _return[i].r_t, _return[i].h_t_1, _W_h, _s_t[i], _b_h);
		_return[i].h_t = forward_h_t(_return[i].z_t, _return[i].h_t_1, _return[i].tilde_h_t);
		_return[i].pred = prediction(_W_y, _return[i].h_t, _b_y);
		_return[i].error = error(_return[i].pred, _target[i]);
		_return[i].mse = calc_mse(_return[i].pred, _target[i]);

		_return[i].p_h_t = middle_PG_W_y(_W_y, _return[i].error);
		if (i < _timesteps - 1)
		{
			_return[i + 1].h_t_1 = _return[i].h_t;
		}
	}
	return _return;
}

std::vector<Forward_Parameter> forward_pass_prediction(const std::vector<double>& _h_t_1, const std::vector<double>& _b_y, const std::vector<std::vector<double>>& _W_y, const size_t& _timesteps, std::vector<std::vector<double>>& _U_r, std::vector<std::vector<double>>& _U_z, std::vector<std::vector<double>>& _U_h, std::vector<std::vector<double>>& _W_r, std::vector<std::vector<double>>& _W_z, std::vector<std::vector<double>>& _W_h, std::vector<double>& _b_r, std::vector<double>& _b_z, std::vector<double>& _b_h, std::vector<std::vector<double>>& _s_t)
{
	std::vector<Forward_Parameter> _return(_timesteps);
	_return[0].h_t_1 = _h_t_1;
	for (size_t i = 0; i < _timesteps; i++)
	{
		_return[i].z_t = forward_z_t(_U_z, _return[i].h_t_1, _W_z, _s_t[i], _b_z);
		_return[i].r_t = forward_r_t(_U_r, _return[i].h_t_1, _W_r, _s_t[i], _b_r);
		_return[i].tilde_h_t = forward_tilde_h_t(_U_h, _return[i].r_t, _return[i].h_t_1, _W_h, _s_t[i], _b_h);
		_return[i].h_t = forward_h_t(_return[i].z_t, _return[i].h_t_1, _return[i].tilde_h_t);
		_return[i].pred = prediction(_W_y, _return[i].h_t, _b_y);
		if (i < _timesteps - 1)
		{
			_return[i + 1].h_t_1 = _return[i].h_t;
		}
	}
	return _return;
}

struct Backward_Parameter
{
	std::vector<double> p_z_t;
	std::vector<double> p_tilde_h_t;
	std::vector<double> p_h_t_1_FHState;
	std::vector<double> p_a_t;
	std::vector<double> p_r_t;
	std::vector<double> p_h_t_1_CHState;
	std::vector<double> p_b_r;
	std::vector<double> p_h_t_1_RState;
	std::vector<double> p_b_z;
	std::vector<double> p_h_t_1_UState;
	std::vector<double> final_gradient;
	std::vector<double> p_h_t_1;
	std::vector<std::vector<double>> p_W_r;
	std::vector<std::vector<double>> p_U_r;
	std::vector<std::vector<double>> p_W_z;
	std::vector<std::vector<double>> p_U_z;
	std::vector<std::vector<double>> p_W_h;
	std::vector<std::vector<double>> p_U_h;
	std::vector<std::vector<double>> p_W_y;

};

std::vector<Backward_Parameter> backward_pass(const size_t& _timesteps, std::vector<Forward_Parameter>& _forward_pass, std::vector<std::vector<double>>& _U_r, std::vector<std::vector<double>>& _U_z, std::vector<std::vector<double>>& _U_h, std::vector<std::vector<double>>& _s_t)
{
	std::vector<Backward_Parameter> _return(_timesteps);
	_return[_timesteps - 1].p_h_t_1 = _forward_pass[_timesteps - 1].p_h_t;
	for (int i = _timesteps - 1; i >= 0; i--)
	{
		_return[i].p_z_t = backward_p_z_t(_return[i].p_h_t_1, _forward_pass[i].tilde_h_t, _forward_pass[i].h_t_1);
		_return[i].p_tilde_h_t = backward_p_tilde_h_t(_return[i].p_h_t_1, _forward_pass[i].z_t);
		_return[i].p_h_t_1_FHState = backward_p_h_t_1_FHState(_return[i].p_h_t_1, _forward_pass[i].z_t);
		_return[i].p_a_t = backward_p_a_t(_return[i].p_tilde_h_t, _forward_pass[i].tilde_h_t);
		_return[i].p_r_t = backward_p_r_t(_U_h, _forward_pass[i].h_t_1, _return[i].p_a_t);
		_return[i].p_h_t_1_CHState = backward_p_h_t_1_CHState(_U_h, _forward_pass[i].r_t, _return[i].p_a_t);
		_return[i].p_b_r = backward_p_b_r(_return[i].p_r_t, _forward_pass[i].r_t);
		_return[i].p_h_t_1_RState = backward_p_h_t_1_RState(_U_r, _return[i].p_b_r);
		_return[i].p_b_z = backward_p_b_z(_return[i].p_z_t, _forward_pass[i].z_t);
		_return[i].p_h_t_1_UState = backward_p_h_t_1_UState(_U_z, _return[i].p_b_z);
		_return[i].final_gradient = sum_gradient(_return[i].p_h_t_1_FHState, _return[i].p_h_t_1_CHState, _return[i].p_h_t_1_RState, _return[i].p_h_t_1_UState);
		_return[i].p_W_r = backward_PG_W_r(_return[i].p_b_r, _s_t[i]);
		_return[i].p_U_r = backward_PG_U_r(_return[i].p_b_r, _forward_pass[i].h_t_1);
		_return[i].p_W_z = backward_PG_W_z(_return[i].p_b_z, _s_t[i]);
		_return[i].p_U_z = backward_PG_U_z(_return[i].p_b_z, _forward_pass[i].h_t_1);
		_return[i].p_W_h = backward_PG_W_h(_return[i].p_a_t, _s_t[i]);
		_return[i].p_U_h = backward_PG_U_h(_return[i].p_a_t, _forward_pass[i].h_t_1, _forward_pass[i].r_t);
		_return[i].p_W_y = backward_PG_W_y(_forward_pass[i].error, _forward_pass[i].h_t);
		if (i > 0)
		{
			_return[i - 1].p_h_t_1 = _return[i].final_gradient;
			for (size_t z = 0; z < _return[i].final_gradient.size(); z++)
			{
				_return[i - 1].p_h_t_1[z] = _return[i].final_gradient[z] + _forward_pass[i - 1].p_h_t[z];
			}
		}
	}
	return _return;
}

template<typename T>
void backward_sum_vector_time(const std::vector<Backward_Parameter>& _backward, const std::vector<T> Backward_Parameter::* _mem, std::vector<T>& _return_vector)
{
	_return_vector.assign((_backward[0].*_mem).size(), 0);
	for (size_t i = 0; i < _backward.size(); i++)
	{
		std::vector<double> _calc_1 = _backward[i].*_mem;
		for (size_t j = 0; j < (_backward[0].*_mem).size(); j++)
		{
			_return_vector[j] = _return_vector[j] + _calc_1[j];
		}
	}
}

template<typename T>
void forward_sum_vector_time(const std::vector<Forward_Parameter>& _forward, const std::vector<T> Forward_Parameter::* _mem, std::vector<T>& _return_vector)
{
	_return_vector.assign((_forward[0].*_mem).size(), 0);
	for (size_t i = 0; i < _forward.size(); i++)
	{
		std::vector<double> _calc_1 = _forward[i].*_mem;
		for (size_t j = 0; j < (_forward[0].*_mem).size(); j++)
		{
			_return_vector[j] = _return_vector[j] + _calc_1[j];
		}
	}
}

template<typename T>
void backward_sum_matrix_time(const std::vector<Backward_Parameter>& _backward, std::vector<std::vector<T>> Backward_Parameter::* _mem, std::vector<std::vector<T>>& _return_matrix)
{
	_return_matrix.assign((_backward[0].*_mem).size(), std::vector<T>((_backward[0].*_mem)[0].size(), 0));
	for (size_t i = 0; i < _backward.size(); i++)
	{
		std::vector<std::vector<double>> _calc_1 = _backward[i].*_mem;
		for (size_t j = 0; j < (_backward[0].*_mem).size(); j++)
		{
			for (size_t v = 0; v < (_backward[0].*_mem)[0].size(); v++)
			{
				_return_matrix[j][v] = _return_matrix[j][v] + _calc_1[j][v];
			}
		}
	}
}

std::vector<std::vector<double>> load_csv(const std::string& _file_name)
{
	std::ifstream _file(_file_name);
	if (!_file)
	{
		std::cout << "File not found" << std::endl;
	}
	std::vector<std::vector<double>> _csv_data;
	std::string _string_1;
	while (std::getline(_file, _string_1))
	{
		std::vector<double> _csv_row(2);
		std::stringstream _sstream(_string_1);
		std::string _string_2;
		std::string _string_3;
		std::getline(_sstream, _string_2, ';');
		std::getline(_sstream, _string_3, ';');
		std::replace(_string_2.begin(), _string_2.end(), ',', '.');
		std::replace(_string_3.begin(), _string_3.end(), ',', '.');
		_csv_row[0] = std::stod(_string_2);
		_csv_row[1] = std::stod(_string_3);
		_csv_data.push_back(_csv_row);
	}
	return _csv_data;
}

void save_csv(const std::string& _file_name, const std::vector<std::vector<double>>& _matrix)
{
	std::ofstream _file(_file_name);
	if (!_file)
	{
		std::cout << "File creation not possible" << std::endl;
	}
	for (size_t i = 0; i < _matrix.size(); i++)
	{
		std::ostringstream _csv_one;
		std::ostringstream _csv_two;
		_csv_one << _matrix[i][0];
		_csv_two << _matrix[i][1];
		std::string _string_1 = _csv_one.str();
		std::string _string_2 = _csv_two.str();
		std::replace(_string_1.begin(), _string_1.end(), '.', ',');
		std::replace(_string_2.begin(), _string_2.end(), '.', ',');
		_file << _string_1 << ';' << _string_2 << '\n';
	}
}

struct Min_Max_Scaler
{
	std::vector<double> min;
	std::vector<double> max;
};

Min_Max_Scaler fit_scaler(const std::vector<std::vector<double>>& _matrix)
{
	Min_Max_Scaler _return_scaler;
	_return_scaler.min.assign(_matrix[0].size(), std::numeric_limits<double>::max());
	_return_scaler.max.assign(_matrix[0].size(), std::numeric_limits<double>::lowest());
	for (size_t i = 0; i < _matrix.size(); i++)
	{
		for (size_t j = 0; j < _matrix[0].size(); j++)
		{	
			if (_matrix[i][j] > _return_scaler.max[j])
			{
				_return_scaler.max[j] = _matrix[i][j];
			}
			if (_matrix[i][j] < _return_scaler.min[j])
			{
				_return_scaler.min[j] = _matrix[i][j];
			}
		}
	}
	return _return_scaler;
}

std::vector<std::vector<double>> min_max_trans(const Min_Max_Scaler& _scaler, const std::vector<std::vector<double>>& _matrix)
{
	std::vector<std::vector<double>> _return_matrix = _matrix;
	for (size_t i = 0; i < _matrix.size(); i++)
	{
		for (size_t j = 0; j < _matrix[i].size(); j++)
		{
			_return_matrix[i][j] = 2.0 * (_matrix[i][j] - _scaler.min[j]) / (_scaler.max[j] - _scaler.min[j]) - 1.0;
		}
	}
	return _return_matrix;
}

std::vector<std::vector<double>> min_max_i_trans(const Min_Max_Scaler& _scaler, const std::vector<std::vector<double>>& _matrix)
{
	std::vector<std::vector<double>> _return_matrix = _matrix;
	for (size_t i = 0; i < _matrix.size(); i++)
	{
		for (size_t j = 0; j < _matrix[i].size(); j++)
		{
			_return_matrix[i][j] = (_matrix[i][j] + 1.0) * (_scaler.max[j] - _scaler.min[j]) / 2.0 + _scaler.min[j];
		}
	}
	return _return_matrix;
}

std::vector<double> calc_total_MSE_error(const std::vector<double>& _error, const double& _steps)
{
	std::vector<double> _return_vector(_error.size());
	for (size_t i = 0; i < _error.size(); i++)
	{
		_return_vector[i] = _error[i] / _steps;
	}
	return _return_vector;
}

std::vector<double> calc_MSE_test_data(const size_t& _timesteps, const std::vector<std::vector<double>>& _back_scaled_test_data, const std::vector<std::vector<double>>& _back_scaled_prediction)
{
	std::vector<double> _test_data_mse = create_zero_vector(2);
	for (size_t k = 0; k < _timesteps; k++)
	{
		for (size_t t = 0; t < 2; t++)
		{
			_test_data_mse[t] = _test_data_mse[t] + (_back_scaled_test_data[k][t] - _back_scaled_prediction[k][t]) * (_back_scaled_test_data[k][t] - _back_scaled_prediction[k][t]);
		}
	}

	for (size_t t = 0; t < 2; t++)
	{
		_test_data_mse[t] = _test_data_mse[t] / _timesteps;
	}
	return _test_data_mse;
}

std::vector<double> calc_RMSE_test_data(const std::vector<double>& _total_MSE)
{
	std::vector<double> _test_data_rmse = create_zero_vector(2);

	for (size_t t = 0; t < 2; t++)
	{
		_test_data_rmse[t] = std::sqrt(_total_MSE[t]);
	}

	return _test_data_rmse;
}

std::vector<double> calc_MAE_test_data(const size_t& _timesteps, const std::vector<std::vector<double>>& _back_scaled_test_data, const std::vector<std::vector<double>>& _back_scaled_prediction)
{
	std::vector<double> _test_data_mae = create_zero_vector(2);
	for (size_t k = 0; k < _timesteps; k++)
	{
		for (size_t t = 0; t < 2; t++)
		{
			_test_data_mae[t] = _test_data_mae[t] + std::abs(_back_scaled_test_data[k][t] - _back_scaled_prediction[k][t]);
		}
	}

	for (size_t t = 0; t < 2; t++)
	{
		_test_data_mae[t] = _test_data_mae[t] / _timesteps;
	}
	return _test_data_mae;
}

void gru_gradient_checking()
{
	size_t timesteps = 2;
	size_t size_input = 2;
	size_t size_hidden = 2;
	size_t size_output = 2;

	double e_value = 0.00001;

	//W_h
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);
			//h_t_1 = create_val_vector(size_hidden, 0.3);

			std::vector<std::vector<double>> W_h_plus = W_h;
			W_h_plus[i][j] = W_h_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h_plus, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> W_h_minus = W_h;
			W_h_minus[i][j] = W_h_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h_minus, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_W_h;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_h, total_W_h);

			std::cout << "W_h: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_W_h[i][j] << " Relative Diff: " << std::abs(grad_num - total_W_h[i][j]) / (std::abs(grad_num) + std::abs(total_W_h[i][j])) << std::endl;
		}
	}
	//W_y
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> W_y_plus = W_y;
			W_y_plus[i][j] = W_y_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y_plus, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> W_y_minus = W_y;
			W_y_minus[i][j] = W_y_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y_minus, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_W_y;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_y, total_W_y);

			std::cout << "W_y: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_W_y[i][j] << " Relative Diff: " << std::abs(grad_num - total_W_y[i][j]) / (std::abs(grad_num) + std::abs(total_W_y[i][j])) << std::endl;
		}
	}
	//W_z
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> W_z_plus = W_z;
			W_z_plus[i][j] = W_z_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z_plus, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> W_z_minus = W_z;
			W_z_minus[i][j] = W_z_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z_minus, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_W_z;

			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_z, total_W_z);

			std::cout << "W_z: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_W_z[i][j] << " Relative Diff: " << std::abs(grad_num - total_W_z[i][j]) / (std::abs(grad_num) + std::abs(total_W_z[i][j])) << std::endl;
		}
	}
	//W_r
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> W_r_plus = W_r;
			W_r_plus[i][j] = W_r_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r_plus, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> W_r_minus = W_r;
			W_r_minus[i][j] = W_r_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r_minus, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);


			std::vector<std::vector<double>> total_W_r;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_r, total_W_r);

			std::cout << "W_r: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_W_r[i][j] << " Relative Diff: " << std::abs(grad_num - total_W_r[i][j]) / (std::abs(grad_num) + std::abs(total_W_r[i][j])) << std::endl;
		}
	}
	//U_r
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> U_r_plus = U_r;
			U_r_plus[i][j] = U_r_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r_plus, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> U_r_minus = U_r;
			U_r_minus[i][j] = U_r_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r_minus, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_U_r;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_r, total_U_r);
	
			std::cout << "U_r: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_U_r[i][j] << " Relative Diff: " << std::abs(grad_num - total_U_r[i][j]) / (std::abs(grad_num) + std::abs(total_U_r[i][j])) << std::endl;
		}
	}
	//U_z
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> U_z_plus = U_z;
			U_z_plus[i][j] = U_z_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z_plus, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> U_z_minus = U_z;
			U_z_minus[i][j] = U_z_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z_minus, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_U_z;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_z, total_U_z);

			std::cout << "U_z: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_U_z[i][j] << " Relative Diff: " << std::abs(grad_num - total_U_z[i][j]) / (std::abs(grad_num) + std::abs(total_U_z[i][j])) << std::endl;
		}
	}
	//U_h
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{

			std::vector<double> h_t_1 = create_zero_vector(size_hidden);
			std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
			std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
			std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
			std::vector<double> b_r = create_zero_vector(size_hidden);
			std::vector<double> b_z = create_zero_vector(size_hidden);
			std::vector<double> b_h = create_zero_vector(size_hidden);
			std::vector<double> b_y = create_zero_vector(size_input);

			std::vector<std::vector<double>> s_t = create_matrix(2, 2);
			std::vector<std::vector<double>> target = create_matrix(2, 2);

			h_t_1 = create_zero_vector(size_hidden);

			std::vector<std::vector<double>> U_h_plus = U_h;
			U_h_plus[i][j] = U_h_plus[i][j] + e_value;

			std::vector<Forward_Parameter> forward_plus;
			forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h_plus, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_plus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_plus[u].error.size(); k++)
				{
					loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
				}
			}

			std::vector<std::vector<double>> U_h_minus = U_h;
			U_h_minus[i][j] = U_h_minus[i][j] - e_value;

			std::vector<Forward_Parameter> forward_minus;
			forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h_minus, W_r, W_z, W_h, b_r, b_z, b_h, s_t);

			double loss_minus = 0.0;

			for (int u = 0; u < timesteps; u++)
			{
				for (int k = 0; k < forward_minus[u].error.size(); k++)
				{
					loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
				}
			}

			double grad_num = (loss_plus - loss_minus) / (e_value * 2);

			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_U_h;
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_h, total_U_h);

			std::cout << "U_h: " << " i: " << i << " j: " << j << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_U_h[i][j] << " Relative Diff: " << std::abs(grad_num - total_U_h[i][j]) / (std::abs(grad_num) + std::abs(total_U_h[i][j])) << std::endl;
		}
	}
	//b_r
	for (size_t i = 0; i < 2; i++)
	{
		std::vector<double> h_t_1 = create_zero_vector(size_hidden);
		std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
		std::vector<double> b_r = create_zero_vector(size_hidden);
		std::vector<double> b_z = create_zero_vector(size_hidden);
		std::vector<double> b_h = create_zero_vector(size_hidden);
		std::vector<double> b_y = create_zero_vector(size_input);

		std::vector<std::vector<double>> s_t = create_matrix(2, 2);
		std::vector<std::vector<double>> target = create_matrix(2, 2);

		h_t_1 = create_zero_vector(size_hidden);

		std::vector<double> b_r_plus = b_r;
		b_r_plus[i] = b_r_plus[i] + e_value;

		std::vector<Forward_Parameter> forward_plus;
		forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r_plus, b_z, b_h, s_t);

		double loss_plus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_plus[u].error.size(); k++)
			{
				loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
			}
		}

		std::vector<double> b_r_minus = b_r;
		b_r_minus[i] = b_r_minus[i] - e_value;

		std::vector<Forward_Parameter> forward_minus;
		forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r_minus, b_z, b_h, s_t);

		double loss_minus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_minus[u].error.size(); k++)
			{
				loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
			}
		}

		double grad_num = (loss_plus - loss_minus) / (e_value * 2);

		std::vector<Forward_Parameter> forward;
		forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
		std::vector<Backward_Parameter> backward;
		backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

		std::vector<double> total_b_r;
		backward_sum_vector_time(backward, &Backward_Parameter::p_b_r, total_b_r);

		std::cout << "b_r: " << " i: " << i << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_b_r[i] << " Relative Diff: " << std::abs(grad_num - total_b_r[i]) / (std::abs(grad_num) + std::abs(total_b_r[i])) << std::endl;
	}
	//b_z
	for (size_t i = 0; i < 2; i++)
	{
		std::vector<double> h_t_1 = create_zero_vector(size_hidden);
		std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
		std::vector<double> b_r = create_zero_vector(size_hidden);
		std::vector<double> b_z = create_zero_vector(size_hidden);
		std::vector<double> b_h = create_zero_vector(size_hidden);
		std::vector<double> b_y = create_zero_vector(size_input);

		std::vector<std::vector<double>> s_t = create_matrix(2, 2);
		std::vector<std::vector<double>> target = create_matrix(2, 2);

		h_t_1 = create_zero_vector(size_hidden);

		std::vector<double> b_z_plus = b_z;
		b_z_plus[i] = b_z_plus[i] + e_value;

		std::vector<Forward_Parameter> forward_plus;
		forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z_plus, b_h, s_t);

		double loss_plus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_plus[u].error.size(); k++)
			{
				loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
			}
		}

		std::vector<double> b_z_minus = b_z;
		b_z_minus[i] = b_z_minus[i] - e_value;

		std::vector<Forward_Parameter> forward_minus;
		forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z_minus, b_h, s_t);

		double loss_minus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_minus[u].error.size(); k++)
			{
				loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
			}
		}

		double grad_num = (loss_plus - loss_minus) / (e_value * 2);

		std::vector<Forward_Parameter> forward;
		forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
		std::vector<Backward_Parameter> backward;
		backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

		std::vector<double> total_b_z;
		backward_sum_vector_time(backward, &Backward_Parameter::p_b_z, total_b_z);

		std::cout << "b_z: " << " i: " << i << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_b_z[i] << " Relative Diff: " << std::abs(grad_num - total_b_z[i]) / (std::abs(grad_num) + std::abs(total_b_z[i])) << std::endl;
	}
	//b_h
	for (size_t i = 0; i < 2; i++)
	{
		std::vector<double> h_t_1 = create_zero_vector(size_hidden);
		std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
		std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
		std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);
		std::vector<double> b_r = create_zero_vector(size_hidden);
		std::vector<double> b_z = create_zero_vector(size_hidden);
		std::vector<double> b_h = create_zero_vector(size_hidden);
		std::vector<double> b_y = create_zero_vector(size_input);

		std::vector<std::vector<double>> s_t = create_matrix(2, 2);
		std::vector<std::vector<double>> target = create_matrix(2, 2);

		h_t_1 = create_zero_vector(size_hidden);

		std::vector<double> b_h_plus = b_h;
		b_h_plus[i] = b_h_plus[i] + e_value;

		std::vector<Forward_Parameter> forward_plus;
		forward_plus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h_plus, s_t);

		double loss_plus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_plus[u].error.size(); k++)
			{
				loss_plus = loss_plus + 0.5 * (forward_plus[u].error[k] * forward_plus[u].error[k]);
			}
		}

		std::vector<double> b_h_minus = b_h;
		b_h_minus[i] = b_h_minus[i] - e_value;

		std::vector<Forward_Parameter> forward_minus;
		forward_minus = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h_minus, s_t);

		double loss_minus = 0.0;

		for (int u = 0; u < timesteps; u++)
		{
			for (int k = 0; k < forward_minus[u].error.size(); k++)
			{
				loss_minus = loss_minus + 0.5 * (forward_minus[u].error[k] * forward_minus[u].error[k]);
			}
		}

		double grad_num = (loss_plus - loss_minus) / (e_value * 2);

		std::vector<Forward_Parameter> forward;
		forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
		std::vector<Backward_Parameter> backward;
		backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

		std::vector<double> total_b_h;
		backward_sum_vector_time(backward, &Backward_Parameter::p_a_t, total_b_h);

		std::cout << "b_h: " << " i: " << i << " Grad_Num - Grad_Calc: " << " Absolute Diff: " << grad_num - total_b_h[i] << " Relative Diff: " << std::abs(grad_num - total_b_h[i]) / (std::abs(grad_num) + std::abs(total_b_h[i])) << std::endl;
	}
}

void gru_example_1()
{
	std::vector<std::vector<double>> csv_matrix = load_csv("MFST_RAW_DATA_CSV.csv");
	std::vector<std::vector<double>> csv_matrix_train(csv_matrix.begin(), csv_matrix.begin() + 1551);
	Min_Max_Scaler scaler = fit_scaler(csv_matrix_train);
	std::vector<std::vector<double>> scaled_csv_matrix = min_max_trans(scaler, csv_matrix_train);

	size_t timesteps = 50;
	size_t size_input = 2;
	size_t size_hidden = 8;
	size_t size_output = 2;

	std::vector<double> h_t_1 = create_zero_vector(size_hidden);
	std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);

	std::vector<double> b_r = create_zero_vector(size_hidden);
	std::vector<double> b_z = create_zero_vector(size_hidden);
	std::vector<double> b_h = create_zero_vector(size_hidden);
	std::vector<double> b_y = create_zero_vector(size_input);

	std::vector<double> MSE_error = create_zero_vector(size_hidden);

	double rating = 0.005;

	int count = 0;

	for (size_t sample = 0; sample <= 1500; sample++)
	{
		std::vector<std::vector<double>> s_t(scaled_csv_matrix.begin() + sample, scaled_csv_matrix.begin() + sample + timesteps);
		std::vector<std::vector<double>> target(scaled_csv_matrix.begin() + sample + 1, scaled_csv_matrix.begin() + sample + timesteps + 1);

		h_t_1 = create_zero_vector(size_hidden);
		for (size_t epoch = 0; epoch < 100; epoch++)
		{
			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_W_r;
			std::vector<std::vector<double>> total_W_z;
			std::vector<std::vector<double>> total_W_h;
			std::vector<std::vector<double>> total_U_r;
			std::vector<std::vector<double>> total_U_z;
			std::vector<std::vector<double>> total_U_h;
			std::vector<std::vector<double>> total_W_y;

			std::vector<double> total_b_r;
			std::vector<double> total_b_z;
			std::vector<double> total_b_h;
			std::vector<double> total_b_y;

			std::vector<double> total_error;

			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_r, total_W_r);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_z, total_W_z);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_h, total_W_h);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_r, total_U_r);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_z, total_U_z);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_h, total_U_h);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_y, total_W_y);

			backward_sum_vector_time(backward, &Backward_Parameter::p_b_r, total_b_r);
			backward_sum_vector_time(backward, &Backward_Parameter::p_b_z, total_b_z);
			backward_sum_vector_time(backward, &Backward_Parameter::p_a_t, total_b_h);
			forward_sum_vector_time(forward, &Forward_Parameter::error, total_b_y);
			forward_sum_vector_time(forward, &Forward_Parameter::mse, total_error);

			W_r = update_matrix(W_r, total_W_r, rating);
			U_r = update_matrix(U_r, total_U_r, rating);
			W_z = update_matrix(W_z, total_W_z, rating);
			U_z = update_matrix(U_z, total_U_z, rating);
			W_h = update_matrix(W_h, total_W_h, rating);
			U_h = update_matrix(U_h, total_U_h, rating);
			W_y = update_matrix(W_y, total_W_y, rating);

			b_r = update_vector(b_r, total_b_r, rating);
			b_h = update_vector(b_h, total_b_h, rating);
			b_z = update_vector(b_z, total_b_z, rating);
			b_y = update_vector(b_y, total_b_y, rating);

			MSE_error = update_vector_MSE(total_error, MSE_error);
			count++;
		}
	}
	std::vector<double> total_MSE_loss = calc_total_MSE_error(MSE_error, count * timesteps);

	std::vector<double> h_t_12 = create_zero_vector(size_hidden);

	std::vector<std::vector<double>> csv_matrix_test(csv_matrix.begin() + 1600, csv_matrix.begin() + 1600 + timesteps);
	std::vector<std::vector<double>> test_data(csv_matrix.begin() + 1600 + 1, csv_matrix.begin() + 1600 + timesteps + 1);
	std::vector<std::vector<double>> scaled_csv_matrix_test = min_max_trans(scaler, csv_matrix_test);
	std::vector<std::vector<double>> s_t2 = scaled_csv_matrix_test;
	std::vector<Forward_Parameter> forward2;
	forward2 = forward_pass_prediction(h_t_12, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t2);

	std::vector<std::vector<double>> prediction_csv = create_matrix(timesteps, size_output);

	for (size_t r = 0; r < timesteps; r++)
	{
		prediction_csv[r] = forward2[r].pred;
	}

	std::vector<std::vector<double>> back_scaled_prediction = min_max_i_trans(scaler, prediction_csv);

	save_csv("Test_CSV.csv", back_scaled_prediction);
	std::cout << "End save" << std::endl;

	std::vector<double> test_data_mse = calc_MSE_test_data(timesteps, test_data, back_scaled_prediction);
	std::vector<double> test_data_rmse = calc_RMSE_test_data(test_data_mse);
	std::vector<double> test_data_mae = calc_MAE_test_data(timesteps, test_data, back_scaled_prediction);
}

void gru_example_2()
{
	size_t timesteps = 3;
	size_t size_input = 2;
	size_t size_hidden = 8;
	size_t size_output = 2;

	std::vector<double> h_t_1 = create_zero_vector(size_hidden);
	std::vector<std::vector<double>> U_r = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> U_z = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> U_h = create_matrix(size_hidden, size_hidden);
	std::vector<std::vector<double>> W_r = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_z = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_h = create_matrix(size_hidden, size_input);
	std::vector<std::vector<double>> W_y = create_matrix(size_input, size_hidden);

	std::vector<double> b_r = create_zero_vector(size_hidden);
	std::vector<double> b_z = create_zero_vector(size_hidden);
	std::vector<double> b_h = create_zero_vector(size_hidden);
	std::vector<double> b_y = create_zero_vector(size_input);

	std::vector<std::vector<double>> s_t = { { 0.1, 0.2 },
											 { 0.2, 0.3 },
											 { 0.3, 0.4 } };

	std::vector<std::vector<double>> target = { { 0.2, 0.3 },
												{ 0.3, 0.4 },
												{ 0.4, 0.5 } };

	double rating = 0.001;

	for (int sample = 0; sample < 200; sample++)
	{
		double x1 = -1.0 + (1.0 + 1.0) * (rand() / (double)RAND_MAX);
		double x2 = -1.0 + (1.0 + 1.0) * (rand() / (double)RAND_MAX);

		s_t[0] = { x1, x2 };
		s_t[1] = { x1, x2 };
		s_t[2] = { x1, x2 };

		target[0] = { x1 + 0.1, x2 + 0.1 };
		target[1] = { x1 + 0.1, x2 + 0.1 };
		target[2] = { x1 + 0.1, x2 + 0.1 };

		h_t_1 = create_zero_vector(size_hidden);

		for (int epoch = 0; epoch < 5000; epoch++)
		{
			std::vector<Forward_Parameter> forward;
			forward = forward_pass(h_t_1, target, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t);
			std::vector<Backward_Parameter> backward;
			backward = backward_pass(timesteps, forward, U_r, U_z, U_h, s_t);

			std::vector<std::vector<double>> total_W_r;
			std::vector<std::vector<double>> total_W_z;
			std::vector<std::vector<double>> total_W_h;
			std::vector<std::vector<double>> total_U_r;
			std::vector<std::vector<double>> total_U_z;
			std::vector<std::vector<double>> total_U_h;
			std::vector<std::vector<double>> total_W_y;

			std::vector<double> total_b_r;
			std::vector<double> total_b_z;
			std::vector<double> total_b_h;
			std::vector<double> total_b_y;

			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_r, total_W_r);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_z, total_W_z);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_h, total_W_h);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_r, total_U_r);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_z, total_U_z);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_U_h, total_U_h);
			backward_sum_matrix_time(backward, &Backward_Parameter::p_W_y, total_W_y);

			backward_sum_vector_time(backward, &Backward_Parameter::p_b_r, total_b_r);
			backward_sum_vector_time(backward, &Backward_Parameter::p_b_z, total_b_z);
			backward_sum_vector_time(backward, &Backward_Parameter::p_a_t, total_b_h);
			forward_sum_vector_time(forward, &Forward_Parameter::error, total_b_y);

			W_r = update_matrix(W_r, total_W_r, rating);
			U_r = update_matrix(U_r, total_U_r, rating);
			W_z = update_matrix(W_z, total_W_z, rating);
			U_z = update_matrix(U_z, total_U_z, rating);
			W_h = update_matrix(W_h, total_W_h, rating);
			U_h = update_matrix(U_h, total_U_h, rating);
			W_y = update_matrix(W_y, total_W_y, rating);

			b_r = update_vector(b_r, total_b_r, rating);
			b_h = update_vector(b_h, total_b_h, rating);
			b_z = update_vector(b_z, total_b_z, rating);
			b_y = update_vector(b_y, total_b_y, rating);
		}
	}
	std::vector<double> h_t_12 = create_zero_vector(size_hidden);
	std::vector<std::vector<double>> s_t2 = { { 0.1, 0.2 },
										      {-0.2, 0.7},
											  {0.3, 0.5} };

	std::vector<Forward_Parameter> forward2;
	forward2 = forward_pass_prediction(h_t_12, b_y, W_y, timesteps, U_r, U_z, U_h, W_r, W_z, W_h, b_r, b_z, b_h, s_t2);
	print_vector(forward2[0].pred);
	print_vector(forward2[1].pred);
	print_vector(forward2[2].pred);
}

void random_walk_drift_example_1()
{
	std::vector<std::vector<double>> csv_matrix = load_csv("MFST_RAW_DATA_CSV.csv");

	std::vector<double> target;
	size_t timesteps = 50;
	std::vector<std::vector<double>> prediction_s_t = create_matrix(timesteps, 2);
	std::vector<std::vector<double>> matrix_target(csv_matrix.begin() + 1600 + 1, csv_matrix.begin() + 1600 + timesteps + 1);

	for (size_t i = 0; i < timesteps; i++)
	{
		std::vector<std::vector<double>> s_t(csv_matrix.begin(), csv_matrix.begin() + 1600 + 1 + i);

		double drift_0 = 0.0;
		for (size_t j = 1; j < s_t.size(); j++)
		{
			drift_0 = drift_0 + s_t[j][0] - s_t[j - 1][0];
		}
		drift_0 = drift_0 / (s_t.size() - 1);
		double prediction_0 = s_t.back()[0] + drift_0;
		prediction_s_t[i][0] = prediction_0;

		double drift_1 = 0.0;
		for (size_t j = 1; j < s_t.size(); j++)
		{
			drift_1 = drift_1 + s_t[j][1] - s_t[j - 1][1];
		}
		drift_1 = drift_1 / (s_t.size() - 1);
		double prediction_1 = s_t.back()[1] + drift_1;
		prediction_s_t[i][1] = prediction_1;
	}

	std::vector<double> test_data_mse = calc_MSE_test_data(timesteps, matrix_target, prediction_s_t);
	std::vector<double> test_data_rmse = calc_RMSE_test_data(test_data_mse);
	std::vector<double> test_data_mae = calc_MAE_test_data(timesteps, matrix_target, prediction_s_t);
}

void random_walk_drift_example_2()
{
	std::vector<double> data = { 0.1, 0.2, 0.3, 0.4 };
	double drift = 0.0;
	double noise = 0.0;

	for (size_t i = 1; i < data.size(); i++)
	{
		drift = drift + data[i] - data[i - 1];
	}
	drift = drift / (data.size() - 1);

	for (size_t j = 1; j <= 3; j++)
	{
		double prediction = data.back() + drift;
		data.push_back(prediction);
	}
	print_vector(data);
}

int main()
{
	//gru_gradient_checking();
	gru_example_1();
	//gru_example_2();

	//random_walk_drift_example_1();
	//random_walk_drift_example_2();
	return 0;
}
