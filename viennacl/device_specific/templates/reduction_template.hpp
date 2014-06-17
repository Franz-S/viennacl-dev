#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_TEMPLATE_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_TEMPLATE_HPP

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file viennacl/generator/scalar_reduction.hpp
 *
 * Kernel template for the scalar reduction operation
*/

#include <vector>

#include "viennacl/backend/opencl.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/tree_parsing/filter.hpp"
#include "viennacl/device_specific/tree_parsing/read_write.hpp"
#include "viennacl/device_specific/tree_parsing/evaluate_expression.hpp"
#include "viennacl/device_specific/utils.hpp"

#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/templates/reduction_utils.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl
{

  namespace device_specific
  {

    class reduction_template : public template_base
    {

    public:
      class parameters : public template_base::parameters
      {
      public:
        parameters(const char * scalartype, unsigned int simd_width,
                   unsigned int group_size, unsigned int num_groups,
                   unsigned int decomposition) : template_base::parameters(scalartype, simd_width, group_size, 1, 2), num_groups_(num_groups), decomposition_(decomposition){ }

        unsigned int num_groups() const { return num_groups_; }
        unsigned int decomposition() const { return decomposition_; }

      private:
        unsigned int num_groups_;
        unsigned int decomposition_;
      };

    private:

      static bool is_reduction(scheduler::statement_node const & node)
      {
        return node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
            || node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE;
      }

      unsigned int lmem_used(unsigned int scalartype_size) const
      {
        return parameters_.local_size_0()*scalartype_size;
      }

      void configure_impl(unsigned int kernel_id, statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & n_arg)  const
      {
        //configure ND range
        if(kernel_id==0)
        {
          kernel.global_work_size(0,parameters_.local_size_0()*parameters_.num_groups());
          kernel.global_work_size(1,1);
        }
        else
        {
          kernel.global_work_size(0,parameters_.local_size_0());
          kernel.global_work_size(1,1);
        }

        //set arguments
        cl_uint size = get_vector_size(statements.data().front());
        kernel.arg(n_arg++, size/parameters_.simd_width());

        std::vector<scheduler::statement_node const *> reductions;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter(&is_reduction, reductions), false);

        unsigned int i = 0;
        unsigned int j = 0;
        for(std::vector<scheduler::statement_node const *>::const_iterator it = reductions.begin() ; it != reductions.end() ; ++it)
        {
          if(tmp_.size() <= i)
            tmp_.push_back(viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, parameters_.num_groups()*utils::scalartype_size(parameters_.scalartype())));
          kernel.arg(n_arg++, tmp_[i]);
          i++;

          if(utils::is_index_reduction(**it))
          {
            if(tmpidx_.size() <= j)
              tmpidx_.push_back(viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, parameters_.num_groups()*4));
            kernel.arg(n_arg++, tmpidx_[j]);
            j++;
          }
        }
      }

      void add_kernel_arguments(statements_container const & statements, std::string & arguments_string) const{
        arguments_string += generate_value_kernel_argument("unsigned int", "N");

        std::vector<scheduler::statement_node const *> reductions;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          tree_parsing::traverse(*it, it->root(), tree_parsing::filter(&is_reduction, reductions), false);

        for(std::vector<scheduler::statement_node const *>::iterator it = reductions.begin() ; it != reductions.end() ; ++it)
        {
          arguments_string += generate_pointer_kernel_argument("__global", parameters_.scalartype(),  "temp" + tools::to_string(std::distance(reductions.begin(), it)));
          if(utils::is_index_reduction(**it))
            arguments_string += generate_pointer_kernel_argument("__global", "unsigned int",  "temp" + tools::to_string(std::distance(reductions.begin(), it)) + "idx");
        }
      }

      void core_0(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & /*mapping*/) const
      {
        unsigned int N = exprs.size();

        std::vector<scheduler::op_element> rops(N);
        std::vector<std::string> accs(N);
        std::vector<std::string> accsidx(N);
        std::vector<std::string> local_buffers_names(N);

        for(unsigned int k = 0 ; k < N ; ++k){
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          }
          else{
            rops[k].type        = root_op.type;
          }
          accs[k] = "acc"+tools::to_string(k);
          accsidx[k] = accs[k] + "idx";
          local_buffers_names[k] = "buf"+tools::to_string(k);
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < N ; ++k){
          stream << parameters_.scalartype() << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << "unsigned int " << accsidx[k] << " = " << 0 << ";" << std::endl;
        }

        std::string init;
        std::string upper_bound;
        std::string inc;
        if(parameters_.decomposition()){
          init = "get_global_id(0)";
          upper_bound = "N";
          inc = "get_global_size(0)";
        }
        else{
          stream << "unsigned int chunk_size = (N + get_num_groups(0)-1)/get_num_groups(0);" << std::endl;
          stream << "unsigned int chunk_start = get_group_id(0)*chunk_size;" << std::endl;
          stream << "unsigned int chunk_end = min(chunk_start+chunk_size, N);" << std::endl;
          init = "chunk_start + get_local_id(0)";
          upper_bound = "chunk_end";
          inc = "get_local_size(0)";
        }

        stream << "for(unsigned int i = " << init << "; i < " << upper_bound << " ; i += " << inc << "){" << std::endl;
        stream.inc_tab();
        {
          //Fetch vector entry
          std::set<std::string>  cache;
          for(std::vector<mapped_scalar_reduction*>::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
          {
            tree_parsing::read_write(tree_parsing::read_write_traversal::FETCH, parameters_.simd_width(), "reg", cache, (*it)->statement(), (*it)->root_idx(), index_tuple("i", "N"),stream,(*it)->mapping(), tree_parsing::PARENT_NODE_TYPE);
          }
          //Update accs;
          for(unsigned int k = 0 ; k < exprs.size() ; ++k)
          {
            viennacl::scheduler::statement const & statement = exprs[k]->statement();
            unsigned int root_idx = exprs[k]->root_idx();
            mapping_type const & mapping = exprs[k]->mapping();
            index_tuple idx("i","N");
            if(parameters_.simd_width() > 1){
              for(unsigned int a = 0 ; a < parameters_.simd_width() ; ++a){
                std::string value = tree_parsing::evaluate_expression(statement,root_idx,idx,a,mapping,tree_parsing::LHS_NODE_TYPE);
                if(statement.array()[root_idx].op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                  value += "*";
                  value += tree_parsing::evaluate_expression(statement,root_idx,idx,a,mapping,tree_parsing::RHS_NODE_TYPE);
                }
                compute_reduction(stream,accsidx[k],"i",accs[k],value,rops[k]);
              }
            }
            else{
              std::string value = tree_parsing::evaluate_expression(statement,root_idx,idx,-1,mapping,tree_parsing::LHS_NODE_TYPE);
              if(statement.array()[root_idx].op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
                value += "*";
                value += tree_parsing::evaluate_expression(statement,root_idx,idx,-1,mapping,tree_parsing::RHS_NODE_TYPE);
              }
              compute_reduction(stream,accsidx[k],"i",accs[k],value,rops[k]);
            }
          }
        }
        stream.dec_tab();
        stream << "}" << std::endl;


        //Declare and fill local memory
        for(unsigned int k = 0 ; k < N ; ++k){
          stream << "__local " << parameters_.scalartype() << " " << local_buffers_names[k] << "[" << parameters_.local_size_0() << "];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << "__local " << "unsigned int" << " " << local_buffers_names[k] << "idx[" << parameters_.local_size_0() << "];" << std::endl;
        }


        for(unsigned int k = 0 ; k < N ; ++k){
          stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << local_buffers_names[k] << "idx[lid] = " << accsidx[k] << ";" << std::endl;
        }

        //Reduce and write to temporary buffers
        reduce_1d_local_memory(stream, parameters_.local_size_0(),local_buffers_names,rops);

        stream << "if(lid==0){" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          stream << "temp"<< k << "[get_group_id(0)] = buf" << k << "[0];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << "temp"<< k << "idx[get_group_id(0)] = buf" << k << "idx[0];" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;
      }


      void core_1(utils::kernel_generation_stream& stream, std::vector<mapped_scalar_reduction*> exprs, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        unsigned int N = exprs.size();
        std::vector<scheduler::op_element> rops(N);
        std::vector<std::string> accs(N);
        std::vector<std::string> accsidx(N);
        std::vector<std::string> local_buffers_names(N);
        for(unsigned int k = 0 ; k < N ; ++k){
          scheduler::op_element root_op = exprs[k]->statement().array()[exprs[k]->root_idx()].op;
          rops[k].type_family = scheduler::OPERATION_BINARY_TYPE_FAMILY;
          if(root_op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE){
            rops[k].type        = scheduler::OPERATION_BINARY_ADD_TYPE;
          }
          else{
            rops[k].type        = root_op.type;
          }
          accs[k] = "acc"+tools::to_string(k);
          accsidx[k] = accs[k] + "idx";
          local_buffers_names[k] = "buf"+tools::to_string(k);
        }

        stream << "unsigned int lid = get_local_id(0);" << std::endl;

        for(unsigned int k = 0 ; k < exprs.size() ; ++k){
          stream << "__local " << parameters_.scalartype() << " " << local_buffers_names[k] << "[" << parameters_.local_size_0() << "];" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << "__local " << "unsigned int" << " " << local_buffers_names[k] << "idx[" << parameters_.local_size_0() << "];" << std::endl;
        }

        for(unsigned int k = 0 ; k < local_buffers_names.size() ; ++k){
          stream << parameters_.scalartype() << " " << accs[k] << " = " << neutral_element(rops[k]) << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << "unsigned int" << " " << accsidx[k] << " = " << 0 << ";" << std::endl;
        }

        stream << "for(unsigned int i = lid ; i < " << parameters_.num_groups() << " ; i += get_local_size(0)){" << std::endl;
        stream.inc_tab();
        for(unsigned int k = 0 ; k < N ; ++k)
          compute_reduction(stream,accsidx[k],"temp"+tools::to_string(k)+"idx[i]",accs[k],"temp"+tools::to_string(k)+"[i]",rops[k]);
        stream.dec_tab();
        stream << "}" << std::endl;

        for(unsigned int k = 0 ; k < local_buffers_names.size() ; ++k)
        {
          stream << local_buffers_names[k] << "[lid] = " << accs[k] << ";" << std::endl;
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            stream << local_buffers_names[k] << "idx[lid] = " << accsidx[k] << ";" << std::endl;
        }


        //Reduce and write final result
        reduce_1d_local_memory(stream, parameters_.local_size_0(),local_buffers_names,rops);
        for(unsigned int k = 0 ; k < N ; ++k)
        {
          std::string suffix = "";
          if(utils::is_index_reduction(exprs[k]->statement().array()[exprs[k]->root_idx()]))
            suffix = "idx";
          exprs[k]->access_name(local_buffers_names[k]+suffix+"[0]");
        }

        stream << "if(lid==0){" << std::endl;
        stream.inc_tab();
        unsigned int i = 0;
        for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
          stream << tree_parsing::evaluate_expression(*it, it->root(), index_tuple("0", "N"), -1, mapping[i++], tree_parsing::PARENT_NODE_TYPE) << ";" << std::endl;

        stream.dec_tab();
        stream << "}" << std::endl;
      }

      cl_uint get_vector_size(viennacl::scheduler::statement const & s) const
      {
        scheduler::statement::container_type exprs = s.array();
        for(scheduler::statement::container_type::iterator it = exprs.begin() ; it != exprs.end() ; ++it)
        {
          if(is_scalar_reduction(*it)){
            scheduler::statement_node const * current_node = &(*it);
            //The LHS of the prod is a vector
            if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
              return utils::call_on_vector(current_node->lhs, utils::internal_size_fun());
            //The LHS of the prod is a vector expression
            current_node = &exprs[current_node->lhs.node_index];
            if(current_node->lhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
              return utils::call_on_vector(current_node->lhs, utils::internal_size_fun());
            if(current_node->rhs.type_family==scheduler::VECTOR_TYPE_FAMILY)
              return utils::call_on_vector(current_node->lhs, utils::internal_size_fun());
          }
        }
        throw "unexpected expression tree";
      }

      void core(unsigned int kernel_id, utils::kernel_generation_stream& stream, statements_container const & statements, std::vector<mapping_type> const & mapping) const
      {
        std::vector<mapped_scalar_reduction*> exprs;
        for(std::vector<mapping_type>::const_iterator it = mapping.begin() ; it != mapping.end() ; ++it)
          for(mapping_type::const_iterator iit = it->begin() ; iit != it->end() ; ++iit)
            if(mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
              exprs.push_back(p);

        if(kernel_id==0)
          core_0(stream,exprs,statements,mapping);
        else
          core_1(stream,exprs,statements,mapping);
      }


    public:
      reduction_template(reduction_template::parameters const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base(parameters, binding_policy), parameters_(parameters){ }

    private:
      reduction_template::parameters const & parameters_;
      mutable std::vector< viennacl::ocl::handle<cl_mem> > tmp_;
      mutable std::vector< viennacl::ocl::handle<cl_mem> > tmpidx_;
    };

  }

}

#endif
