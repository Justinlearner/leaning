// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
// This function sets attributes for each variable/equation in the app. The
// attributes are set via standardized function calls. The first parameter for
// each function call is the variable index (starting at zero). The first set of
// variable/equation attributes are the variable name (any string), the variable
// type (SCALAR/VECTOR), and the equation type (EXPLICIT_TIME_DEPENDENT/
// TIME_INDEPENDENT/AUXILIARY). The next set of attributes describe the
// dependencies for the governing equation on the values and derivatives of the
// other variables for the value term and gradient term of the RHS and the LHS.
// The final pair of attributes determine whether a variable represents a field
// that can nucleate and whether the value of the field is needed for nucleation
// rate calculations.

void
variableAttributeLoader::loadVariableAttributes()
{
  // Variable 0
  set_variable_name(0, "c");
  set_variable_type(0, SCALAR);
  set_variable_equation_type(0, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(0, "c");
  set_dependencies_gradient_term_RHS(0, "c, grad(c), n, grad(n)");

  set_allowed_to_nucleate(0, false);
  set_need_value_nucleation(0, true);

  // Variable 1
  set_variable_name(1, "phi");
  set_variable_type(1, SCALAR);
  set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(1, "phi,c,xi");
  set_dependencies_gradient_term_RHS(1, "");

  set_allowed_to_nucleate(1, true);
  set_need_value_nucleation(1, true);

  // Variable 2
  set_variable_name(2, "xi");
  set_variable_type(2, SCALAR);
  set_variable_equation_type(2, AUXILIARY);

  set_dependencies_value_term_RHS(2, "");
  set_dependencies_gradient_term_RHS(2, "grad(phi)");
}

// =============================================================================================
// explicitEquationRHS (needed only if one or more equation is explict time
// dependent)
// =============================================================================================
// This function calculates the right-hand-side of the explicit time-dependent
// equations for each variable. It takes "variable_list" as an input, which is a
// list of the value and derivatives of each of the variables at a specific
// quadrature point. The (x,y,z) location of that quadrature point is given by
// "q_point_loc". The function outputs two terms to variable_list -- one
// proportional to the test function and one proportional to the gradient of the
// test function. The index for each variable in this list corresponds to the
// index given at the top of this file.

template <int dim, int degree>
void
customPDE<dim, degree>::explicitEquationRHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc) const
{
  // --- Getting the values and derivatives of the model variables ---

  // The dimensionless solute supersaturation and its derivatives
  scalarvalueType c  = variable_list.get_scalar_value(0);
  scalargradType  cx = variable_list.get_scalar_gradient(0);

  // The order parameter and its derivatives
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // The auxiliary parameter and its derivatives
  scalarvalueType xi = variable_list.get_scalar_value(2);

  // --- Setting the expressions for the terms in the governing equations ---

  // Interpolation function and its derivative
  scalarvalueType pV  = 3.0 * phi * phi - 2.0 * phi * phi * phi;
  scalarvalueType pnV = 6.0 * phi - 6.0 * phi * phi;

  // KKS model c_alpha and c_beta as a function of c and h
  scalarvalueType c_alpha =
    (B2 * (c - cbtmin * pV) + A2 * calmin * pV) / (A2 * pV + B2 * (1.0 - pV));
  scalarvalueType c_beta = (A2 * (c - calmin * (1.0 - pV)) + B2 * cbtmin * (1.0 - pV)) /
                           (A2 * pV + B2 * (1.0 - pV));

  // Free energy for each phase and their first and second derivatives
  scalarvalueType faV   = A0 + A2 * (c_alpha - calmin) * (c_alpha - calmin);
  scalarvalueType facV  = 2.0 * A2 * (c_alpha - calmin);
  scalarvalueType faccV = constV(2.0) * A2;
  scalarvalueType fbV   = B0 + B2 * (c_beta - cbtmin) * (c_beta - cbtmin);
  scalarvalueType fbcV  = 2.0 * B2 * (c_beta - cbtmin);
  scalarvalueType fbccV = constV(2.0) * B2;

  // Double-Well function (can be used to tune the interfacial energy)
  scalarvalueType fbarrierV  = phi * phi - 2.0 * phi * phi * phi + phi * phi * phi * phi;
  scalarvalueType fbarriernV = 2.0 * phi - 6.0 * phi * phi + 4.0 * phi * phi * phi;

  // Calculation of interface normal vector
  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = phix / (normgradn + constV(regval));

  // The cosine of theta
  scalarvalueType cth = normal[0];
  // The sine of theta
  scalarvalueType sth = normal[1];
  // The cosine of 4 theta
  scalarvalueType c4th =
    sth * sth * sth * sth + cth * cth * cth * cth - constV(6.0) * sth * sth * cth * cth;

  // Anisotropic term
  scalarvalueType a_n;
  // a_n = (constV(1.0)+constV(epsilon)*std::cos(constV(4.0)*(theta)));
  a_n = (constV(1.0) + constV(epsilon) * c4th);

  // -------------------------------------------------
  // Nucleation expressions
  // -------------------------------------------------
  dealii::VectorizedArray<double> source_term = constV(0.0);
  dealii::VectorizedArray<double> gamma       = constV(1.0);
  seedNucleus(q_point_loc, source_term, gamma);
  // -------------------------------------------------

  // Set the terms in the governing equations

  // For concentration
  scalarvalueType eq_c = c;
  scalargradType  eqx_c =
    constV(-McV * userInputs.dtValue) * (cx + (c_alpha - c_beta) * pnV * phix);
  // For order parameter (gamma is a variable order parameter mobility factor)
  scalarvalueType eq_phi =
    phi - constV(userInputs.dtValue * MnV) * gamma *
          ((fbV - faV) * pnV - (c_beta - c_alpha) * fbcV * pnV + H_barrier * fbarriernV);

  // --- Submitting the terms for the governing equations ---

  // Terms for the equation to evolve the concentration
  variable_list.set_scalar_value_term_RHS(0, eq_c);
  variable_list.set_scalar_gradient_term_RHS(0, eqx_c);

  // Terms for the equation to evolve the order parameter
  variable_list.set_scalar_value_term_RHS(1, eq_phi + source_term);
}

// =================================================================================
// seedNucleus: a function particular to this app
// =================================================================================
template <int dim, int degree>
void
customPDE<dim, degree>::seedNucleus(
  const dealii::Point<dim, dealii::VectorizedArray<double>> &q_point_loc,
  dealii::VectorizedArray<double>                           &source_term,
  dealii::VectorizedArray<double>                           &gamma) const
{
  for (typename std::vector<nucleus<dim>>::const_iterator thisNucleus =
         this->nuclei.begin();
       thisNucleus != this->nuclei.end();
       ++thisNucleus)
    {
      if (thisNucleus->seededTime + thisNucleus->seedingTime > this->currentTime)
        {
          // Calculate the weighted distance function to the order parameter
          // freeze boundary (weighted_dist = 1.0 on that boundary)
          dealii::VectorizedArray<double> weighted_dist =
            this->weightedDistanceFromNucleusCenter(
              thisNucleus->center,
              userInputs.get_nucleus_freeze_semiaxes(thisNucleus->orderParameterIndex),
              q_point_loc,
              thisNucleus->orderParameterIndex);

          for (unsigned i = 0; i < gamma.size(); i++)
            {
              if (weighted_dist[i] <= 1.0)
                {
                  gamma[i] = 0.0;

                  // Seed a nucleus if it was added to the list of nuclei this
                  // time step
                  if (thisNucleus->seedingTimestep == this->currentIncrement)
                    {
                      // Find the weighted distance to the outer edge of the
                      // nucleus and use it to calculate the order parameter
                      // source term
                      dealii::Point<dim, double> q_point_loc_element;
                      for (unsigned int j = 0; j < dim; j++)
                        {
                          q_point_loc_element(j) = q_point_loc(j)[i];
                        }
                      double r = this->weightedDistanceFromNucleusCenter(
                        thisNucleus->center,
                        userInputs.get_nucleus_semiaxes(thisNucleus->orderParameterIndex),
                        q_point_loc_element,
                        thisNucleus->orderParameterIndex);

                      double avg_semiaxis = 0.0;
                      for (unsigned int j = 0; j < dim; j++)
                        {
                          avg_semiaxis += thisNucleus->semiaxes[j];
                        }
                      avg_semiaxis /= dim;

                      source_term[i] =
                        0.5 *
                        (1.0 - std::tanh(avg_semiaxis * (r - 1.0) / interface_coeff));
                    }
                }
            }
        }
    }
}
// =============================================================================================
// nonExplicitEquationRHS (needed only if one or more equation is time
// independent or auxiliary)
// =============================================================================================
// This function calculates the right-hand-side of all of the equations that are
// not explicit time-dependent equations. It takes "variable_list" as an input,
// which is a list of the value and derivatives of each of the variables at a
// specific quadrature point. The (x,y,z) location of that quadrature point is
// given by "q_point_loc". The function outputs two terms to variable_list --
// one proportional to the test function and one proportional to the gradient of
// the test function. The index for each variable in this list corresponds to
// the index given at the top of this file.

template <int dim, int degree>
void
customPDE<dim, degree>::nonExplicitEquationRHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc) const
{
  // --- Getting the values and derivatives of the model variables ---

  // The order parameter and its derivatives
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // --- Setting the expressions for the terms in the governing equations ---

  // The azimuthal angle
  // scalarvalueType theta;
  // for (unsigned i=0; i< phi.size();i++){
  //	theta[i] = std::atan2(phix[1][i],phix[0][i]);
  //}

  // Calculation of interface normal vector
  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = phix / (normgradn + constV(regval));

  // The cosine of theta
  scalarvalueType cth = normal[0];
  // The sine of theta
  scalarvalueType sth = normal[1];

  // The cosine of 4 theta
  scalarvalueType c4th =
    sth * sth * sth * sth + cth * cth * cth * cth - constV(6.0) * sth * sth * cth * cth;
  // The sine of 4 theta
  scalarvalueType s4th =
    constV(4.0) * sth * cth * cth * cth - constV(4.0) * sth * sth * sth * cth;

  // Anisotropic term
  scalarvalueType a_n;
  // a_n = (constV(1.0)+constV(epsilon)*std::cos(constV(4.0)*(theta)));
  a_n = (constV(1.0) + constV(epsilon) * c4th);

  // gradient energy coefficient, its derivative and square
  scalarvalueType a_d;
  // a_d = constV(-4.0)*constV(epsilon)*std::sin(constV(4.0)*(theta));
  a_d = constV(-4.0) * constV(epsilon) * s4th;

  // The anisotropy term that enters in to the equation for xi
  scalargradType aniso;
  aniso[0] = a_n * a_n * phix[0] - a_n * a_d * phix[1];
  aniso[1] = a_n * a_n * phix[1] + a_n * a_d * phix[0];

  // Define the terms in the equations

  scalargradType eqx_xi = (-aniso);

  // --- Submitting the terms for the governing equations ---

  variable_list.set_scalar_gradient_term_RHS(2, eqx_xi);
}

// =============================================================================================
// equationLHS (needed only if at least one equation is time independent)
// =============================================================================================
// This function calculates the left-hand-side of time-independent equations. It
// takes "variable_list" as an input, which is a list of the value and
// derivatives of each of the variables at a specific quadrature point. The
// (x,y,z) location of that quadrature point is given by "q_point_loc". The
// function outputs two terms to variable_list -- one proportional to the test
// function and one proportional to the gradient of the test function -- for the
// left-hand-side of the equation. The index for each variable in this list
// corresponds to the index given at the top of this file. If there are multiple
// elliptic equations, conditional statements should be sed to ensure that the
// correct residual is being submitted. The index of the field being solved can
// be accessed by "this->currentFieldIndex".

template <int dim, int degree>
void
customPDE<dim, degree>::equationLHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc) const
{}
