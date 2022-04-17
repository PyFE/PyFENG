function [w_res, wn_arr] = sol_w(types, order)
%% define variables and equations
assert(ismember(types,{'density','AD'}));
syms a b x x0 t sigma m u
w0 = 1/sigma^2 * (a*b*(exp(-x)-exp(-x0)) + (x-x0)*(a+1/2*sigma^2));
mu = a*(b-exp(x))/exp(x) - 1/2 * sigma^2;
wn_arr = [w0];
w_res = w0;
%% loop for each Wn
for n = 2:order+1
    disp(n);
    lamda_1 = 1/2 * sigma^2 * diff(wn_arr(n-1), x, 2) -...
        mu*diff(wn_arr(n-1), x);
    % f = 'diff(wn_arr(m), x)*diff(wn_arr(n-m), x)';
    lamda_2 = get_sum(wn_arr, 1, n);
    if ( strcmp(types, 'AD') )
        lamda = lamda_1 - 1/2*sigma^2 * lamda_2 + ...
            (n-1<2)*(exp(x) + diff(mu, x));
    else
        lamda = lamda_1 - 1/2*sigma^2 * lamda_2 + (n-1<2)*diff(mu, x);
    end
    f2 = u^(n-2)*subs(lamda, x, x0+(x-x0)*u);
    wn_arr = [wn_arr; int(f2, u, 0, 1)];
    w_res = w_res + wn_arr(n)*t^(n-1);
end
end

function lamda = get_sum(wn_arr,k0, kn)
%% get the sum of cross-product of partitial wn
syms x
lamda = 0;
for m = k0:kn-1
    lamda = lamda + diff(wn_arr(m), x)*diff(wn_arr(kn-m), x);
end
end
% disp(wn_arr);
% disp(w_res);