function [V,Y,i,nrmrestotnew,s,RES] = rational_ksm_shifted_systems(A1,Bdiag,C,params)
% function [V,Y,it,nrmrestotnew,s] = rational_ksm_shifted_systems(A1,Bdiag,C,params)
%      
% Approximately Solve
%                A1 X  +  X (diag(Bdiag) \otimes I_k) = C*D^T
%     X \approx V*Y
%
% by the minimal residual Rational Krylov subspace method 
%
% Input:
% A1               coeff. matrix.  n x n
% Bdiag               shifts
%
% params.m        max space dimension allowed
% params.tol      Algebraic stopping tolerance 
% params.shift_init  initial pole for basis construction
% params.iterative  0: direct solution of the linear systems
%                                  with A1 (default parameter)
%                   1: iterative solution of the linear systems
%                                  with A1 by gmres(50) preconditioned by ILU
%       
% Output:
% V, Y:  low-rank factors of the solution V*Y
% it: number of performed iterations
% nrmrestotnew: residual norm history
% s: vector of poles employed in the basis construction
% RES: mxl matrix whose (i,j) entry is the relative residual norm related
%      related to the j-th shifted linear system at the i-th iteration
%
% Reference:
% Minimal Residual Rational Krylov Subspace Method for Sequences of Shifted Linear Systems
% Hussam Al Daas and Davide Palitta 
% ArXiv: 2507.00267
%
% Please, acknowledge our work by citing our manuscript whenever you use the software 
% provided here in your research.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
% OTHER DEALINGS IN THE SOFTWARE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



m = params.m;
tol = params.tol;
selected_inds = 1;
snew = Bdiag(1);
s = snew;
if ~isfield(params,'iterative')
    params.iterative = 0;
end
if ~isfield(params,'mgs')
    mgs = 1;
else
    mgs = 0;
end

n = size(A1,1);
l = length(Bdiag);
C = full(C);
p = size(C,2);
H = zeros((m+1)*p,m*p);
K = zeros((m+1)*p,m*p);
Y = zeros((m+1)*p,l);
index_noncov = 1 : l;
nrmrestotnew = [];
res = [];

if p == 1
    rr = norm(C);
    V(:,1) = C/rr;
else
    [V(:,1:p),rr] = qr(C,0);
end
Cm = eye(p*(m+1),p)*rr;
nrm_rhs = norm(rr,'fro')*ones(1,l);

for i = 1 : m
    ind = (i-1)*p + 1 : i*p;
    mind = max(ind);
    % rho * mu != nu * eta
    % B1 = nu * A1 - mu * speye(n);
    % B2 = rho * A1 - eta * speye(n);
    % w = B1 \ (B2 * V(:,ind));
    nu = 1;
    mu = -snew;
    rho = 0;
    eta = -1;
    B1 = A1 + snew * speye(n);
    if params.iterative    
        [LL,UU] = ilu(B1);
        w=gmres(B1,V(:,ind),50,tol/10,100,LL,UU);
    else
        % B2 = speye(n);
        w = B1\(V(:,ind)); 
    end 


    % orthogonalization
    for j = 1 : 2
        if mgs == 1
            for k = 1 : mind
                coef = V(:,k)' * w;
                w = w - V(:,k) * coef;
                H(k,ind) = H(k,ind) + coef;
            end
        else
            coef = V(:,1:mind)' * w;
            w = w - V(:,1:mind) * coef;
            H(1:mind,ind) = H(1:mind,ind) + coef;
        end
    end
    [V(:,ind+p),H(ind+p,ind)] = qr(w,'econ');

    % Update of K and H
    ej = zeros(mind+p,p); ej(ind,:) = eye(p);
    K(1:mind+p,ind) = (nu * H(1:mind+p,ind) - rho * ej);
    H(1:mind+p,ind) = mu * H(1:mind+p,ind) - eta * ej;

    % solution of the projected least squares problems
    for t = index_noncov
        temp_H = H(1:mind+p,1:mind) + Bdiag(t) * K(1:mind+p,1:mind);
        [Q,R] = qr(temp_H);
        rhs = Q' * (Cm(1:mind+p,:));
        y = R(1:mind, 1:mind)\rhs(1:mind,:);
        res(t) = norm(rhs(mind+1:mind+p,:),'fro');

        Y(1:mind+p, (t-1)*p+1 : t*p) = K(1:mind+p,1:mind) * y;
    end
    
    % residual norm check
    res=res./nrm_rhs;
    RES(:,i)=res;
    index_noncov = find(res>tol);
    [nrmresnew,index_shift] = max(res);
    selected_inds = ([selected_inds, index_shift]);
    nrmrestotnew = [nrmrestotnew, nrmresnew];

    if (nrmresnew < tol)
        break
    end
    % choose next pole
    snew = Bdiag(index_shift);
    s = [s,snew];
end
V=V(:,1:mind+p);
Y=Y(1:mind+p,:);
