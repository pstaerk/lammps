/* ---------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: Sep/22/2014
   Zhenxing Wang(KU)
------------------------------------------------------------------------- */

#include "fix_gen_conp.h"
#include "atom.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "math.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "stddef.h"
#include "stdlib.h"
#include "string.h"
#include "update.h"
#include "variable.h"

#include "./Eigen/Eigen/Dense"
#include "comm.h"
#include "domain.h"
#include "iostream"
#include "kspace.h"
#include "math_const.h"
#include "mpi.h"
#include "neigh_list.h"
#include "pair.h"

#define EWALD_F 1.12837917
#define EWALD_P 0.3275911
#define A1 0.254829592
#define A2 -0.284496736
#define A3 1.421413741
#define A4 -1.453152027
#define A5 1.061405429

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum { CONSTANT, EQUAL, ATOM };

extern "C" {
void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv, int *info);
void dgetri_(const int *N, double *A, const int *lda, const int *ipiv, double *work,
             const int *lwork, int *info);
}
/* ---------------------------------------------------------------------- */

FixGenConp::FixGenConp(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR, "Illegal fix conp command");
  maxiter = 100;
  tolerance = 0.000001;
  everynum = utils::numeric(FLERR, arg[3], false, lmp);
  eta = utils::numeric(FLERR, arg[4], false, lmp);
  molidL = utils::inumeric(FLERR, arg[5], false, lmp);
  molidR = utils::inumeric(FLERR, arg[6], false, lmp);
  vL = utils::numeric(FLERR, arg[7], false, lmp);
  vR = utils::numeric(FLERR, arg[8], false, lmp);
  if (strcmp(arg[9], "cg") == 0) {
    minimizer = 0;
  } else if (strcmp(arg[9], "inv") == 0) {
    minimizer = 1;
  } else
    error->all(FLERR, "Unknown minimization method");

  outf = fopen(arg[10], "w");
  if (narg == 12) {
    outa = NULL;
    a_matrix_fp = fopen(arg[11], "r");
    if (a_matrix_fp == NULL) error->all(FLERR, "Cannot open A matrix file");
    if (strcmp(arg[11], "org") == 0) {
      a_matrix_f = 1;
    } else if (strcmp(arg[11], "inv") == 0) {
      a_matrix_f = 2;
    } else {
      /* error->all(FLERR,"Unknown A matrix type"); */
      a_matrix_f = 2;
    }
  } else {
    a_matrix_f = 0;
  }
  nr_electrode_atoms = nr_electrode_atoms_old = 0;
  csk = snk = NULL;
  id_to_electrode_id = global_electrode_ids = curr_id_to_electrode_id = local_electrode_ids = NULL;
  Btime = cgtime = Ctime = Ktime = 0;
  runstage = 0;    // after operation
                   // 0:init; 1: a_cal; 2: first sin/cos cal; 3: inv only, aaa inverse
}

/* ---------------------------------------------------------------------- */

FixGenConp::~FixGenConp()
{
  fclose(outf);
  memory->destroy3d_offset(cs, -kmax_created);
  memory->destroy3d_offset(sn, -kmax_created);
  aaa_all.clear();
  bbb_all.clear();
  delete[] curr_id_to_electrode_id;
  delete[] id_to_electrode_id;
  delete[] global_electrode_ids;
  delete[] local_electrode_ids;
  delete[] kxvecs;
  delete[] kyvecs;
  delete[] kzvecs;
  delete[] ug;
  delete[] sfacrl;
  delete[] sfacim;
  delete[] sfacrl_all;
  delete[] sfacim_all;
}

/* ---------------------------------------------------------------------- */

int FixGenConp::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGenConp::init()
{
  MPI_Comm_rank(world, &me);
}

/* ---------------------------------------------------------------------- */

void FixGenConp::setup(int vflag)
{
  g_ewald = force->kspace->g_ewald;
  slab_volfactor = force->kspace->slab_volfactor;
  double accuracy = force->kspace->accuracy;

  int i;
  double qsqsum = 0.0;
  for (i = 0; i < atom->nlocal; i++) { qsqsum += atom->q[i] * atom->q[i]; }
  double tmp, q2;
  MPI_Allreduce(&qsqsum, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsqsum = tmp;
  q2 = qsqsum * force->qqrd2e / force->dielectric;

  // Copied from ewald.cpp
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd * slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0 * MY_PI / xprd;
  unitk[1] = 2.0 * MY_PI / yprd;
  unitk[2] = 2.0 * MY_PI / zprd_slab;

  bigint natoms = atom->natoms;
  double err;
  kxmax = 1;
  kymax = 1;
  kzmax = 1;

  err = rms(kxmax, xprd, natoms, q2);
  while (err > accuracy) {
    kxmax++;
    err = rms(kxmax, xprd, natoms, q2);
  }

  err = rms(kymax, yprd, natoms, q2);
  while (err > accuracy) {
    kymax++;
    err = rms(kymax, yprd, natoms, q2);
  }

  err = rms(kzmax, zprd_slab, natoms, q2);
  while (err > accuracy) {
    kzmax++;
    err = rms(kzmax, zprd_slab, natoms, q2);
  }

  kmax = MAX(kxmax, kymax);
  kmax = MAX(kmax, kzmax);
  kmax3d = 4 * kmax * kmax * kmax + 6 * kmax * kmax + 3 * kmax;

  // My calculation of ks
  for (i = 0; i < kmax; ++i) {
    if (i <= kxmax) ks_x.push_back(i * unitk[0]);
  }

  for (i = -kmax; i < kmax; ++i) {
    if (i <= kymax) ks_y.push_back(i * unitk[1]);
    if (i <= kzmax) ks_z.push_back(i * unitk[2]);
  }

  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  ug = new double[kmax3d];

  double gsqxmx = unitk[0] * unitk[0] * kxmax * kxmax;
  double gsqymx = unitk[1] * unitk[1] * kymax * kymax;
  double gsqzmx = unitk[2] * unitk[2] * kzmax * kzmax;
  gsqmx = MAX(gsqxmx, gsqymx);
  gsqmx = MAX(gsqmx, gsqzmx);

  gsqmx *= 1.00001;

  coeffs();
  kmax_created = kmax;

  // copied from ewald.cpp end

  int nmax = atom->nmax;
  double evscale = 0.069447;
  vL *= evscale;
  vR *= evscale;

  memory->create3d_offset(cs, -kmax, kmax, 3, nmax, "fixconp:cs");
  memory->create3d_offset(sn, -kmax, kmax, 3, nmax, "fixconp:sn");
  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  id_to_electrode_id = new int[natoms + 1];
  curr_id_to_electrode_id = new int[natoms + 1];
  if (runstage == 0) {
    int i;
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++) {
      if (electrode_check(i)) ++nr_electrode_atoms;
    }
    MPI_Allreduce(&nr_electrode_atoms, &nr_electrode_atoms_all, 1, MPI_INT, MPI_SUM, world);

    global_electrode_ids = new int[nr_electrode_atoms_all];
    /* aaa_all = new double[nr_electrode_atoms_all*nr_electrode_atoms_all]; */
    aaa_all = std::vector<double>(nr_electrode_atoms * nr_electrode_atoms_all);
    bbb_all = std::vector<double>(nr_electrode_atoms);
    /* bbb_all = new double[nr_electrode_atoms_all]; */
    local_electrode_ids = new int[nr_electrode_atoms];
    for (i = 0; i < natoms + 1; i++) id_to_electrode_id[i] = -1;
    for (i = 0; i < natoms + 1; i++) curr_id_to_electrode_id[i] = -1;
    if (minimizer == 0) { eleallq = new double[nr_electrode_atoms_all]; }
    if (a_matrix_f == 0) {
      if (me == 0) outa = fopen("amatrix", "w");
      a_cal();
    } else {
      a_read();
    }
    runstage = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixGenConp::pre_force(int vflag)
{
  if (update->ntimestep % everynum == 0) {
    if (strstr(update->integrate_style, "verlet")) {    // not respa
      Btime1 = MPI_Wtime();
      b_cal();
      Btime2 = MPI_Wtime();
      Btime += Btime2 - Btime1;
      if (update->laststep == update->ntimestep) {
        double Btime_all;
        MPI_Reduce(&Btime, &Btime_all, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        double Ctime_all;
        MPI_Reduce(&Ctime, &Ctime_all, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        double Ktime_all;
        MPI_Reduce(&Ktime, &Ktime_all, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        if (me == 0) {
          Btime = Btime_all / comm->nprocs;
          Ctime = Ctime_all / comm->nprocs;
          Ktime = Ktime_all / comm->nprocs;
          fprintf(outf, "B vector calculation time = %g\n", Btime);
          fprintf(outf, "Coulomb calculation time = %g\n", Ctime);
          fprintf(outf, "Kspace calculation time = %g\n", Ktime);
        }
      }
    }
    equation_solve();
    update_charge();
  }
  force_cal(vflag);
}

/* ---------------------------------------------------------------------- */

int FixGenConp::electrode_check(int atomid)
{
  int *molid = atom->molecule;
  if (molid[atomid] == molidL)
    return 1;
  else if (molid[atomid] == molidR)
    return -1;
  else
    return 0;
}

/* ----------------------------------------------------------------------*/
double FixGenConp::b_component(const double &R_x, const double &R_y, const double &R_z,
                               const double &Q_i, const double &alpha, const double &eta,
                               const double &kmax)
{

  double bvec_result = 0.;
  // First, calculate the kspace sums
  for (auto kx : ks_x) {
    for (auto ky : ks_y) {
      for (auto kz : ks_z) {
        // Skalar product of k and R:
        double kR = R_x * kx + R_y * ky + R_z * kz;
        double sinkR = sin(kR);
        double coskR = cos(kR);
        double ksq = kx * kx + ky * ky + kz * kz;

        bvec_result +=
            (sinkR * sinkR * Q_i + coskR * coskR * Q_i) * exp(-.25 * ksq / (alpha * alpha)) / ksq;
      }
    }
  }

  return bvec_result;
}

/* ----------------------------------------------------------------------*/

double FixGenConp::ion_sum(const double &R_x, const double &R_y, const double &R_z, double &alpha)
{
  // Calculate second sum over all atoms temporary variables needed in each
  // iteration
  int inum = force->pair->list->inum;
  int *ilist = force->pair->list->ilist;
  double **position = atom->x;
  double *q = atom->q;

  double r_x, r_y, r_z, absrR;
  double sum = 0.;
  int atom_index;

  // Loop over all atoms
  for (int i = 0; i < inum; i++) {
    atom_index = ilist[i];

    // Ignore atoms if they are not part of the electrode
    if (electrode_check(atom_index) || q[atom_index] == 0.) continue;

    r_x = position[atom_index][0];
    r_y = position[atom_index][1];
    r_z = position[atom_index][2];

    // Norm of R-r
    absrR = sqrt((R_x - r_x) * (R_x - r_x) + (R_y - r_y) * (R_y - r_y) + (R_z - r_z) * (R_z - r_z));
    sum += q[atom_index] * (erfc(alpha * absrR) - erfc(eta * absrR)) / absrR;
  }

  // Return the result of the sum
  return sum;
}

void FixGenConp::b_cal()
{
  // TODO check slab correction
  // Get all the system atoms positions (non-electrode)
  /* std::vector<double> localIonPositions; */
  /* localIonPositions.resize(atom->nlocal * 3); */
  /* double **localIonPositions; */

  double **x = atom->x;
  double *q = atom->q;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int i, j;

  // Accumulate all the localIonPositons
  /* MPI_Allgatherv(localIonPositions, nlocal, MPI_DOUBLE, localIonPositions, 1, 0, MPI_DOUBLE, world); */

  // How many local atoms of the electrode ?
  int local_electrode_nr = 0;
  for (i = 0; i < nlocal; i++)
    if (electrode_check(i)) local_electrode_nr++;

  // Set local potential difference
  std::vector<double> local_b(local_electrode_nr);
  std::vector<int> electrode_ids;
  j = 0;
  for (int i = 0; i < nlocal; i++) {
    if (electrode_check(i) == 1) {
      local_b[j] = vL;
      j++;
      // Add id to the electrode ids
      electrode_ids.push_back(i);
    }
    if (electrode_check(i) == -1) {
      local_b[j] = vR;
      j++;
      electrode_ids.push_back(i);
    }
  }

  // Then, calculate the kspace sum per electrode atom
  double R_x, R_y, R_z, Q_i;
  /* double alpha = 1 / g_ewald; */
  double alpha = g_ewald;
  x = atom->x;

  // Keep track of which parameter we want to write
  int k = 0;
  for (int i : electrode_ids) {
    R_x = x[i][0];
    R_y = x[i][1];
    R_z = x[i][2];
    Q_i = q[i];

    local_b[k] += b_component(R_x, R_y, R_z, Q_i, alpha, eta, kmax);
    local_b[k] += ion_sum(R_x, R_y, R_z, alpha);
    k++;
  }

  // TODO take care to introduce the slab correction for partially periodic
  // systems

  // Take care to correctly store that in bbb_all (copied from previous version)
  // nr_electrode_atoms_list and displs for gathering ele tag list and bbb
  int nprocs = comm->nprocs;
  int nr_electrode_atoms_list[nprocs];
  MPI_Allgather(&nr_electrode_atoms, 1, MPI_INT, nr_electrode_atoms_list, 1, MPI_INT, world);
  int displs[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (i = 1; i < nprocs; ++i) {
    displssum += nr_electrode_atoms_list[i - 1];
    displs[i] = displssum;
  }
  int ele_taglist_all[nr_electrode_atoms_all];
  int tagi;
  MPI_Allgatherv(local_electrode_ids, nr_electrode_atoms, MPI_INT, &ele_taglist_all,
                 nr_electrode_atoms_list, displs, MPI_INT, world);
  for (int i = 0; i < nr_electrode_atoms_all; i++) {
    tagi = ele_taglist_all[i];
    curr_id_to_electrode_id[tagi] = i;
  }

  // gather b to bbb_all and sort in the same order as aaa_all
  double bbb_buf[nr_electrode_atoms_all];
  MPI_Allgatherv(local_b.data(), nr_electrode_atoms, MPI_DOUBLE, &bbb_buf, nr_electrode_atoms_list,
                 displs, MPI_DOUBLE, world);
  int elei;
  for (int i = 0; i < nr_electrode_atoms_all; i++) {
    tagi = global_electrode_ids[i];
    elei = curr_id_to_electrode_id[tagi];
    bbb_all[i] = bbb_buf[elei];
  }
}

/*----------------------------------------------------------------------- */
void FixGenConp::equation_solve()
{
  // solve equations
  if (minimizer == 0) {
    cgtime1 = MPI_Wtime();
    cg();
    cgtime2 = MPI_Wtime();
    cgtime += cgtime2 - cgtime1;
    if (update->laststep == update->ntimestep) {
      double cgtime_all;
      MPI_Reduce(&cgtime, &cgtime_all, 1, MPI_DOUBLE, MPI_SUM, 0, world);
      if (me == 0) {
        cgtime = cgtime_all / comm->nprocs;
        if (screen) fprintf(screen, "conjugate gradient solver time = %g\n", cgtime);
        if (logfile) fprintf(logfile, "conjugate gradient solver time = %g\n", cgtime);
      }
    }
  } else if (minimizer == 1) {
    inv();
  }
}

/*----------------------------------------------------------------------- */
void FixGenConp::a_read()
{
  int i = 0;
  int idx1d;
  if (me == 0) {
    int maxchar = 21 * nr_electrode_atoms_all + 1;
    char line[maxchar];
    char *word;
    while (fgets(line, maxchar, a_matrix_fp) != NULL) {
      word = strtok(line, " \t");
      while (word != NULL) {
        if (i < nr_electrode_atoms_all) {
          global_electrode_ids[i] = atoi(word);
        } else {
          idx1d = i - nr_electrode_atoms_all;
          aaa_all[idx1d] = atof(word);
        }
        word = strtok(NULL, " \t");
        i++;
      }
    }
    fclose(a_matrix_fp);
  }
  MPI_Bcast(global_electrode_ids, nr_electrode_atoms_all, MPI_INT, 0, world);
  MPI_Bcast(aaa_all.data(), nr_electrode_atoms_all * nr_electrode_atoms_all, MPI_DOUBLE, 0, world);

  int tagi;
  for (i = 0; i < nr_electrode_atoms_all; i++) {
    tagi = global_electrode_ids[i];
    id_to_electrode_id[tagi] = i;
  }
}

double FixGenConp::offdiag(const double R_x, const double R_y, const double R_z, const double V,
                           const double alpha, const double eta, const double &prefactor)
{

  double offdiag_result = 0.;
  double ksqr, Rnorm;

  for (auto kx : ks_x) {
    // Find how to ignore the one summation in the middle
    for (auto ky : ks_y) {
      for (auto kz : ks_z) {
        // Skip the k = 0 part (primed sum)
        if (kx == 0 || ky == 0 || kz == 0) continue;

        // Pre-calculations
        ksqr = kx * kx + ky * ky + kz * kz;
        Rnorm = sqrt(R_x * R_x + R_y * R_y + R_z * R_z);

        offdiag_result +=
            exp(-.25 * ksqr / (alpha * alpha)) * cos(R_x * kx + R_y * ky + R_z * kz) / (ksqr);
      }
    }
  }

  // Add the 8pi/V factor
  offdiag_result *= prefactor;

  // Second Summand here
  offdiag_result += (erfc(alpha * Rnorm) - erfc((1.0 / 2.0) * M_SQRT2 * eta * Rnorm)) / Rnorm;
  return offdiag_result;
}

void FixGenConp::write_matrix()
{
  int idx;
  FILE *outa = fopen("debug_matrix", "w");

  // Print the electrode ids
  for (int i = 0; i < nr_electrode_atoms_all; i++) {
    if (i == 0 && me == 0) fprintf(outa, " ");
    if (me == 0) fprintf(outa, "%20d", global_electrode_ids[i]);
  }

  fprintf(outa, "\n");

  for (int i = 0; i < nr_electrode_atoms_all; ++i) {
    for (int j = 0; j < nr_electrode_atoms_all; ++j) {
      idx = i * nr_electrode_atoms_all + j;
      if (j != 0) fprintf(outa, " ");

      fprintf(outa, "%20.12f", aaa_all[idx]);
    }
    fprintf(outa, "\n");
  }
  fclose(outa);
}

/*----------------------------------------------------------------------- */
void FixGenConp::a_cal()
{
  double t1, t2;
  t1 = MPI_Wtime();
  Ktime1 = MPI_Wtime();
  if (me == 0) { fprintf(outf, "A matrix calculating ...\n"); }

  int nprocs = comm->nprocs;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  int i, j, k;
  // List of the number of electrode atoms in each local process:
  int nr_electrode_atoms_list[nprocs];
  MPI_Allgather(&nr_electrode_atoms, 1, MPI_INT, nr_electrode_atoms_list, 1, MPI_INT, world);

  // Create lists of the displacements that we need to use when aggregating per atom properties
  int displs[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (i = 1; i < nprocs; ++i) {
    displssum += nr_electrode_atoms_list[i - 1];
    displs[i] = displssum;
  }

  // Create a one-to one correspondence of the electrode atom ids and the local ordering
  // of the ids
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      // Store what id the jth electrode atom has
      local_electrode_ids[j] = tag[i];
      j++;
    }
  }

  // Gather these id lists
  MPI_Allgatherv(local_electrode_ids, nr_electrode_atoms, MPI_INT, global_electrode_ids,
                 nr_electrode_atoms_list, displs, MPI_INT, world);

  // Aggregate the R_ij's
  double **x = atom->x;
  double *local_position_list = new double[3 * nr_electrode_atoms];
  double *global_position_list = new double[3 * nr_electrode_atoms_all];
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      local_position_list[j] = x[i][0];
      j++;
      local_position_list[j] = x[i][1];
      j++;
      local_position_list[j] = x[i][2];
      j++;
    }
  }

  // Same displacement list as above, only here for vector valued stuff, i.e. with x, y, z components
  int displs2[nprocs];
  int nr_electrode_atoms_list2[nprocs];
  for (i = 0; i < nprocs; i++) {
    nr_electrode_atoms_list2[i] = nr_electrode_atoms_list[i] * 3;
    displs2[i] = displs[i] * 3;
  }

  MPI_Allgatherv(local_position_list, nr_electrode_atoms * 3, MPI_DOUBLE, global_position_list,
                 nr_electrode_atoms_list2, displs2, MPI_DOUBLE, world);

  // Precalculate the constant parts
  double eightPIOverV = 8 * MY_PI / volume;
  /* double alpha = 1 / g_ewald; */
  double alpha = g_ewald;

  double diagonal_add = -M_SQRT2 * alpha + M_SQRT2 * eta / MY_PIS;
  double xx, yy, zz;
  // calculate the matrix entries, locally...
  for (i = 0; i < nr_electrode_atoms_all; ++i) {
    for (j = 0; j < nr_electrode_atoms_all; ++j) {

      xx = global_position_list[(i * j + j) / 3], yy = global_position_list[(i * j + j) / 3 + 1],
      zz = global_position_list[(i * j + j) / 3 + 2];

      // this should be the same as the idx1d in the original code.
      aaa_all[i * nr_electrode_atoms_all + j] =
          offdiag(xx, yy, zz, volume, alpha, eta, eightPIOverV);

      if (i == j) aaa_all[i * nr_electrode_atoms_all + j] += diagonal_add;
    }
  }

  t2 = MPI_Wtime();
  double tsum = t2 - t1;
  double tsum_all;

  MPI_Allreduce(&tsum, &tsum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  if (me == 0) { fprintf(outf, "A matrix calculation time  = %g\n", tsum); }
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2 - Ktime1;

  if (me == 0) write_matrix();
}

/* ---------------------------------------------------------------------- */
void FixGenConp::cg()
{
  int iter, i, j, idx1d;
  double d, beta, ptap, lresnorm, resnorm, netcharge, tmp;
  double res[nr_electrode_atoms_all], p[nr_electrode_atoms_all], ap[nr_electrode_atoms_all];
  for (i = 0; i < nr_electrode_atoms_all; i++) eleallq[i] = 0.0;
  lresnorm = 0.0;
  for (i = 0; i < nr_electrode_atoms_all; ++i) {
    res[i] = bbb_all[i];
    p[i] = bbb_all[i];
    for (j = 0; j < nr_electrode_atoms_all; ++j) {
      idx1d = i * nr_electrode_atoms_all + j;
      tmp = aaa_all[idx1d] * eleallq[j];
      res[i] -= tmp;
      p[i] -= tmp;
    }
    lresnorm += res[i] * res[i];
  }
  for (iter = 1; iter < maxiter; ++iter) {
    d = 0.0;
    for (i = 0; i < nr_electrode_atoms_all; ++i) {
      ap[i] = 0.0;
      for (j = 0; j < nr_electrode_atoms_all; ++j) {
        idx1d = i * nr_electrode_atoms_all + j;
        ap[i] += aaa_all[idx1d] * p[j];
      }
    }
    ptap = 0.0;
    for (i = 0; i < nr_electrode_atoms_all; ++i) { ptap += ap[i] * p[i]; }
    d = lresnorm / ptap;
    resnorm = 0.0;
    for (i = 0; i < nr_electrode_atoms_all; ++i) {
      eleallq[i] = eleallq[i] + d * p[i];
      res[i] = res[i] - d * ap[i];
      resnorm += res[i] * res[i];
    }
    if (resnorm / nr_electrode_atoms_all < tolerance) {
      netcharge = 0.0;
      for (i = 0; i < nr_electrode_atoms_all; ++i) netcharge += eleallq[i];
      if (me == 0) {
        fprintf(outf, "***** Converged at iteration %d. res = %g netcharge = %g\n", iter, resnorm,
                netcharge);
      }
      break;
    }
    beta = resnorm / lresnorm;
    for (i = 0; i < nr_electrode_atoms_all; ++i) { p[i] = res[i] + beta * p[i]; }
    lresnorm = resnorm;
    if (me == 0) { fprintf(outf, "Iteration %d: res = %g\n", iter, lresnorm); }
  }
}
/* ---------------------------------------------------------------------- */
void FixGenConp::inv()
{
  int i, j, k, idx1d;
  if (runstage == 2 && a_matrix_f < 2) {
    int n = nr_electrode_atoms_all;
    /* int n = nr_electrode_atoms_all; */
    /* int lda = nr_electrode_atoms_all; */
    /* int *ipiv = new int[nr_electrode_atoms_all + 1]; */
    /* int lwork = nr_electrode_atoms_all * nr_electrode_atoms_all; */
    /* double *work = new double[lwork]; */
    /* int info; */
    /* int infosum; */

    // Cast to Eigen matrix
    Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(aaa_all.data(), n, n);
    Eigen::MatrixXd inv = m.inverse();

    // Cast back into the vector form
    // TODO check if we should use row major
    inv = inv.reshaped();
    for (int i = 0; i < nr_electrode_atoms_all * nr_electrode_atoms_all; ++i) {
      aaa_all[i] = inv(i);
    }

    /* dgetrf_(&m, &n, aaa_all.data(), &lda, ipiv, &info); */
    /* infosum = info; */
    /* dgetri_(&n, aaa_all.data(), &lda, ipiv, work, &lwork, &info); */
    /* infosum += info; */
    /* delete[] ipiv; */
    /* ipiv = NULL; */
    /* delete[] work; */
    /* work = NULL; */

    /* if (infosum != 0) error->all(FLERR, "Inversion failed!"); */
    /* if (me == 0) { */
    /*   FILE *outinva = fopen("inv_a_matrix", "w"); */
    /*   for (i = 0; i < nr_electrode_atoms_all; i++) { */
    /*     if (i == 0) fprintf(outinva, " "); */
    /*     fprintf(outinva, "%12d", global_electrode_ids[i]); */
    /*   } */
    /*   fprintf(outinva, "\n"); */
    /*   for (k = 0; k < nr_electrode_atoms_all * nr_electrode_atoms_all; k++) { */
    /*     if (k % nr_electrode_atoms_all != 0) { fprintf(outinva, " "); } */
    /*     fprintf(outinva, "%20.10f", aaa_all[k]); */
    /*     if ((k + 1) % nr_electrode_atoms_all == 0) { fprintf(outinva, "\n"); } */
    /*   } */
    /*   fclose(outinva); */
    /* } */
  }
  if (runstage == 2) runstage = 3;
}

/* ---------------------------------------------------------------------- */
void FixGenConp::update_charge()
{
  int i, j, idx1d;
  int elealli, tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int nall = atom->nlocal + atom->nghost;
  double *q = atom->q;
  double **x = atom->x;
  // TODO maybe recast this all as a matrix vector product using Eigen
  for (i = 0; i < nall; ++i) {
    if (electrode_check(i)) {
      tagi = tag[i];
      elealli = id_to_electrode_id[tagi];
      if (minimizer == 0) {
        q[i] = eleallq[elealli];
      } else if (minimizer == 1) {
        eleallq_i = 0.0;
        for (j = 0; j < nr_electrode_atoms_all; j++) {
          idx1d = elealli * nr_electrode_atoms_all + j;
          eleallq_i += aaa_all[idx1d] * bbb_all[j];
        }
        q[i] = eleallq_i;
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
void FixGenConp::force_cal(int vflag)
{
  int i;
  if (force->kspace->energy) {
    double eleqsqsum = 0.0;
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++) {
      if (electrode_check(i)) { eleqsqsum += atom->q[i] * atom->q[i]; }
    }
    double tmp;
    MPI_Allreduce(&eleqsqsum, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
    eleqsqsum = tmp;
    double scale = 1.0;
    double qscale = force->qqrd2e * scale;
    force->kspace->energy += qscale * eta * eleqsqsum / (sqrt(2) * MY_PIS);
  }
  coul_cal(0, NULL, NULL);
}
/* ---------------------------------------------------------------------- */
void FixGenConp::coul_cal(int coulcalflag, double *m, int *local_electrode_ids)
{
  Ctime1 = MPI_Wtime();
  // coulcalflag = 2: a_cal; 1: b_cal; 0: force_cal
  int i, j, k, ii, jj, jnum, itype, jtype, idx1d;
  int checksum, elei, elej, elealli, eleallj;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz;
  double r, r2inv, rsq, grij, etarij, expm2, t, erfc, dudq;
  double forcecoul, ecoul, prefactor, fpair;

  int inum = force->pair->list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = force->pair->list->ilist;
  int *jlist;
  int *numneigh = force->pair->list->numneigh;
  int **firstneigh = force->pair->list->firstneigh;

  double qqrd2e = force->qqrd2e;
  double **cutsq = force->pair->cutsq;
  int itmp;
  double *p_cut_coul = (double *) force->pair->extract("cut_coul", itmp);
  double cut_coulsq = (*p_cut_coul) * (*p_cut_coul);
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = atomtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      checksum = abs(electrode_check(i)) + abs(electrode_check(j));
      if (checksum == 1 || checksum == 2) {
        if (coulcalflag == 0 || checksum == coulcalflag) {
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;
          jtype = atomtype[j];
          if (rsq < cutsq[itype][jtype]) {
            r2inv = 1.0 / rsq;
            if (rsq < cut_coulsq) {
              dudq = 0.0;
              r = sqrt(rsq);
              if (coulcalflag != 0) {
                grij = g_ewald * r;
                expm2 = exp(-grij * grij);
                t = 1.0 / (1.0 + EWALD_P * grij);
                erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                dudq = erfc / r;
              }
              if (checksum == 1)
                etarij = eta * r;
              else if (checksum == 2)
                etarij = eta * r / sqrt(2);
              expm2 = exp(-etarij * etarij);
              t = 1.0 / (1.0 + EWALD_P * etarij);
              erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;

              if (coulcalflag == 0) {
                prefactor = qqrd2e * qtmp * q[j] / r;
                forcecoul = -prefactor * (erfc + EWALD_F * etarij * expm2);
                fpair = forcecoul * r2inv;
                f[i][0] += delx * forcecoul;
                f[i][1] += dely * forcecoul;
                f[i][2] += delz * forcecoul;
                if (newton_pair || j < nlocal) {
                  f[j][0] -= delx * forcecoul;
                  f[j][1] -= dely * forcecoul;
                  f[j][2] -= delz * forcecoul;
                }
                ecoul = -prefactor * erfc;
                force->pair->ev_tally(i, j, nlocal, newton_pair, 0, ecoul, fpair, delx, dely,
                                      delz);    // evdwl=0
              } else {
                dudq -= erfc / r;
                elei = -1;
                elej = -1;
                for (k = 0; k < nr_electrode_atoms; ++k) {
                  if (i < nlocal) {
                    if (local_electrode_ids[k] == tag[i]) {
                      elei = k;
                      if (coulcalflag == 1) {
                        m[k] -= q[j] * dudq;
                        break;
                      }
                    }
                  }
                  if (j < nlocal) {
                    if (local_electrode_ids[k] == tag[j]) {
                      elej = k;
                      if (coulcalflag == 1) {
                        m[k] -= q[i] * dudq;
                        break;
                      }
                    }
                  }
                }
                if (coulcalflag == 2 && checksum == 2) {
                  elealli = id_to_electrode_id[tag[i]];
                  eleallj = id_to_electrode_id[tag[j]];
                  if (elei != -1) {
                    idx1d = elei * nr_electrode_atoms_all + eleallj;
                    m[idx1d] += dudq;
                  }
                  if (elej != -1) {
                    idx1d = elej * nr_electrode_atoms_all + elealli;
                    m[idx1d] += dudq;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  Ctime2 = MPI_Wtime();
  Ctime += Ctime2 - Ctime1;
}

/* ---------------------------------------------------------------------- */
double FixGenConp::rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0 * q2 * g_ewald / prd * sqrt(1.0 / (MY_PI * km * natoms)) *
      exp(-MY_PI * MY_PI * km * km / (g_ewald * g_ewald * prd * prd));
  return value;
}

/* ---------------------------------------------------------------------- */
void FixGenConp::coeffs()
{
  int k, l, m;
  double sqk;

  double g_ewald_sq_inv = 1.0 / (g_ewald * g_ewald);
  double preu = 4.0 * MY_PI / volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m * unitk[0]) * (m * unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
    sqk = (m * unitk[1]) * (m * unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
    sqk = (m * unitk[2]) * (m * unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0] * k) * (unitk[0] * k) + (unitk[1] * l) * (unitk[1] * l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
        ;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1] * l) * (unitk[1] * l) + (unitk[2] * m) * (unitk[2] * m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0] * k) * (unitk[0] * k) + (unitk[2] * m) * (unitk[2] * m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0] * k) * (unitk[0] * k) + (unitk[1] * l) * (unitk[1] * l) +
            (unitk[2] * m) * (unitk[2] * m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;
        }
      }
    }
  }
}
