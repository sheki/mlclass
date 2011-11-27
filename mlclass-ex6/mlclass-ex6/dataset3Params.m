function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_opt = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
sigma_opt = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
err = bitmax;
for c_t=C_opt
    for sigma_t=sigma_opt 
        model= svmTrain(X, y, c_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t));
        predictions = svmPredict(model, Xval);
        err_t =   mean(double(predictions ~= yval));
        if err_t < err
            err = err_t;
            C = c_t;
            sigma = sigma_t;
        end
    end
end
disp(C);
disp(sigma);
% =========================================================================

end
