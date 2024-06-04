
function pdf_value = multivariateGaussianPDF(x)
 % Calculate mean and covariance
 xMean = mean(x, 1);
 xCov = cov(x);
 
 n = length(x); % Dimensionality
 
 % PDF calculation
 coef = 1 / ((2 * pi)^(n/2));
 xDiff = (x - xMean)';
 invxCov = pinv(xCov);
 
 term1 = det(xCov)^0.5;
 term2 = -0.5 * (xDiff);
 term3 = invxCov * xDiff;
 
 expo = term2 .* term3;
 
 pdf_value = coef * term1 * exp(expo);
end