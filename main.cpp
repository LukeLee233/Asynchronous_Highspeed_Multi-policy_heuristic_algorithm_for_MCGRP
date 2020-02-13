#include <iostream>
#include <sys/timeb.h>
#include "file.h"
#include "utils.h"
#include "MCGRP.h"
#include "RNG.h"
#include "NeighborSearch.h"
#include "Memetic.h"
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include "ConstructPolicy.h"
#include "SingleInsert.h"
#include "DoubleInsert.h"
#include "Slice.h"
#include "Extraction.h"
#include "Swap.h"
#include "Invert.h"
#include "TwoOpt.h"
#include <boost/program_options.hpp>

namespace  bpo = boost::program_options;

using namespace std;

struct timeb epoch_start_time;

int main(int argc, char *argv[])
{
    /*-------------------------parse command line----------------------------*/
    string instance_directory;

    int pool_size;
    int evolve_steps;
    int epoch;
    int random_seed;
    int neighbor_size;
    double QNDF_weights;

    bpo::options_description opts("all options");
    bpo::variables_map vm;

    opts.add_options()
        ("help", "MCGRP program")
        ("directory,dir", bpo::value<std::string>(&instance_directory), "the instance directory")
        ("pool_size,ps", bpo::value<int>(&pool_size)->default_value(5), "size of population")
        ("evolve_step,es", bpo::value<int>(&evolve_steps)->default_value(5), "number of evolving step")
        ("epoch,ep", bpo::value<int>(&epoch)->default_value(1))
        ("random_seed,rs", bpo::value<int>(&random_seed)->default_value(0), "number of evolving step")
        ("neighbor_size,ns", bpo::value<int>(&neighbor_size)->default_value(25), "neighbor size of the local search")
        ("qndf_weights,qs", bpo::value<double>(&QNDF_weights)->default_value(0.6), "weight of solution distance")
        ;

    bpo::store(bpo::parse_command_line(argc, argv, opts), vm);
    bpo::notify(vm);

    if(vm.count("help") ){
        std::cout << opts << std::endl;
        return 0;
    }


    if(instance_directory.empty()){
        std::cerr<<"you need to specify a directory!\n";
        return -1;
    }
    if(random_seed<0 || random_seed > 100){
        std::cerr<<"Incorrect random seed base address!\n";
        return -1;
    }
    /*----------------------------------------------------------------*/


    /*----------------------Create search info log----------------------*/
    //create log floder
    string log_dir = instance_directory + "/log";
    boost::filesystem::path log_dir_path(log_dir);
    if (!(boost::filesystem::exists(log_dir_path))) {
        cout << "Create log directory..." << endl;
        if (boost::filesystem::create_directory(log_dir_path)) {
            cout << "Log Directory created successfullly!\n\n";
        }
    }
    else {
        cout << "Log directory already exists!\n\n";
    }


//      create result folder
    boost::posix_time::ptime time_stamp = boost::posix_time::second_clock::local_time();
    stringstream buffer;
    string date_;
    buffer << time_stamp;
    date_ = buffer.str();
    string date_folder = instance_directory + "/log/" + date_;
    boost::filesystem::path date_dir_path(date_folder);
    if (!(boost::filesystem::exists(date_dir_path))) {
        cout << "Create" << date_ << " directory..." << endl;
        if (boost::filesystem::create_directory(date_dir_path)) {
            cout << "Directory created successfullly!\n\n";
        }
    }
    else {
        cout << "Result directory already exists!\n\n";
    }

    //      create result folder
    string result_dir = date_folder + "/result";
    boost::filesystem::path result_dir_path(result_dir);
    if (!(boost::filesystem::exists(result_dir_path))) {
        cout << "Create result directory..." << endl;
        if (boost::filesystem::create_directory(result_dir_path)) {
            cout << "Directory created successfullly!\n\n";
        }
    }
    else {
        cout << "Result directory already exists!\n\n";
    }

/*----------------------Create info recorder----------------------*/
//  record searching parameters
    log_out.open(date_folder + "/parameters.txt", ios::out);
    print(cout, log_out, "Parameters settings:");
    print(cout, log_out, "directory " + instance_directory);
    print(cout, log_out, "base address of seeds: " + to_string(random_seed));
    print(cout, log_out, "search times: " + to_string(epoch));
    print(cout, log_out, "evolution times: " + to_string(evolve_steps));
    print(cout, log_out, "pool size: " + to_string(pool_size));
    print(cout, log_out, "neighbor_size: " + to_string(neighbor_size));
    print(cout, log_out, "QNDF weights: " + to_string(QNDF_weights));
    log_out.close();
/*----------------------------------------------------------------*/


    vector<string> file_set = read_directory(instance_directory);
    for (auto file_name : file_set) {
        cout << "\n\n------------------------Start instance: " << file_name << "---------------------------------\n\n";

        /* global info */
        double bestobj = numeric_limits<decltype(bestobj)>::max();
        double best_solution_time = numeric_limits<decltype(best_solution_time)>::max();
        int best_epoch = -1;
        vector<int> best_buffer;
        vector<double> fitnessVec;
        vector<double> BesttimeVec;
        vector<double> SearchtimeVec;

        /************************************************************************************************/
        /* initialize the instance object */
        instance_num_information instance_info;
        GetTasksNum(instance_directory + '/' + file_name, instance_info);

        RNG rng = RNG();
        MCGRP Mixed_Instance(instance_info, rng);

        Mixed_Instance.load_file_info(instance_directory + '/' + file_name, instance_info);
        Mixed_Instance.create_neighbor_lists(neighbor_size);



        /* initialize the instance object */
        /*----------------------------------------------------------*/
        //construct heuristic for starting point
//        print(log_out, to_string((int) NBS.ns_indi.total_cost));
//        NBS.beta = 0;
//        print(log_out, to_string((int) NBS.ns_indi.total_cost)+','+ to_string((int) NBS.ns_indi.total_vio_load)+ ','+ to_string((double) NBS.beta)+','+
//        to_string((double) NBS.ns_indi.total_cost + NBS.beta*NBS.ns_indi.total_vio_load));
        /*----------------------------------------------------------*/

        log_out.open(date_folder + '/' + file_name + ".log", ios::out);

        for (int start_seed = random_seed; start_seed < random_seed + epoch; start_seed++) {
            cout << "------------------------Start " << start_seed - random_seed + 1
                 << "th times search------------------------\n";

            Mixed_Instance._rng.change(seed[start_seed]);
            Mixed_Instance.best_total_route_length = numeric_limits<double>::max();
            Mixed_Instance.best_sol_time = numeric_limits<double>::max();

            HighSpeedNeighBorSearch NBS(Mixed_Instance);
//            NeighBorSearch NBS(Mixed_Instance);

            ftime(&epoch_start_time);        //针对每轮搜索时间进行测试

            /*----------------------------------------------------------*/
//            //unit test on nearest scanning
//            //        {
//            //            Individual indi;
//            //            nearest_scanning(Mixed_Instance,indi);
//            //            indi;
//            //        }
//            /*----------------------------------------------------------*/
//
//            //unit test on local operator
//            /*----------------------------------------------------------*/
//            {
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//
//
//                //        vector<int> test{-14,11,7,21,-17,6,8,13,16,22,-3,5,4,10,15,12,-20,19,9,18};
//    //        vector<int> res = get_delimeter_coding(test);
//    //        NBS.unpack_seq(res, Mixed_Instance);
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                cout << "Begin Single insert...\n";
//
//                for (int i = 1; i <= 10000; i++) {
//                    if (i % 1000 == 0) {
//                        cout << "Starting Point: " << NBS.cur_solution_cost << endl;
//                        ftime(&cur_time);
//                        cout << "Finish " << i << "th search, spent: "
//                             << (cur_time.time - epoch_start_time.time)
//                                 + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                    }
//                    SingleInsert::unit_test(NBS, Mixed_Instance);
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//                cout<<"Total actual search step: "<<NBS.search_step<<endl;
//                cout<<"Hit rate: "<<(double(NBS.search_step)/1000.0)*100.0<<"%"<<endl;
//            }
            /*----------------------------------------------------------*/

            /*----------------------------------------------------------*/
//            {
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                cout << "Begin Double insert...\n";
//
//                for (int i = 1; i <= 10000; i++) {
//                    if (i % 1000 == 0) {
//                        cout << "Starting Point: " << NBS.cur_solution_cost << endl;
//                        ftime(&cur_time);
//                        cout << "Finish " << i << "th search, spent: "
//                             << (cur_time.time - epoch_start_time.time)
//                                 + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                    }
//                    DoubleInsert::unit_test(NBS, Mixed_Instance);
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//                cout<<"Total actual search step: "<<NBS.search_step<<endl;
//                cout<<"Hit rate: "<<(double(NBS.search_step)/10000.0)*100.0<<"%"<<endl;
//
//            }
            /*----------------------------------------------------------*/


            /*----------------------------------------------------------*/
//            {
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                cout << "Begin Swap...\n";
//
//                for (int i = 1; i <= 10000; i++) {
//                    if (i % 1000 == 0) {
//                        cout << "Starting Point: " << NBS.cur_solution_cost << endl;
//                        ftime(&cur_time);
//                        cout << "Finish " << i << "th search, spent: "
//                             << (cur_time.time - epoch_start_time.time)
//                                 + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                    }
//                    Swap::unit_test(NBS, Mixed_Instance);
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//                cout<<"Total actual search step: "<<NBS.search_step<<endl;
//                cout<<"Hit rate: "<<(double(NBS.search_step)/10000.0)*100.0<<"%"<<endl;
//            }
            /*----------------------------------------------------------*/


            /*----------------------------------------------------------*/
//            {
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//
//                cout << "Begin Invert...\n";
//                for (int i = 1; i <= 10000; i++) {
//                    if (i % 1000 == 0) {
//                        cout << "Starting Point: " << NBS.cur_solution_cost << endl;
//                        ftime(&cur_time);
//                        cout << "Finish " << i << "th search, spent: "
//                             << (cur_time.time - epoch_start_time.time)
//                                 + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                    }
//                    Invert::unit_test(NBS, Mixed_Instance);
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//                cout<<"Total actual search step: "<<NBS.search_step<<endl;
//                cout<<"Hit rate: "<<(double(NBS.search_step)/10000.0)*100.0<<"%"<<endl;
//            }
            /*----------------------------------------------------------*/


            /*----------------------------------------------------------*/
//            {
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                cout << "Begin Two-opt insert...\n";
//
//                for (int i = 1; i <= 10000; i++) {
//                    if (i % 1000 == 0) {
//                        cout << "Starting Point: " << NBS.cur_solution_cost << endl;
//                        ftime(&cur_time);
//                        cout << "Finish " << i << "th search, spent: "
//                             << (cur_time.time - epoch_start_time.time)
//                                 + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                    }
//                    TwoOpt::unit_test(NBS, Mixed_Instance);
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//                cout<<"Total actual search step: "<<NBS.search_step<<endl;
//                cout<<"Hit rate: "<<(double(NBS.search_step)/10000.0)*100.0<<"%"<<endl;
//            }
            /*----------------------------------------------------------*/

//            /*----------------------------------------------------------*/
////            High Speed Local Search
//            {
//                /*------------------------------construct a start point-------------------------------*/
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//                NBS.unpack_seq(initial_solution.sequence, Mixed_Instance);
//                Mixed_Instance.check_best_solution(initial_solution.total_cost,get_negative_coding(initial_solution.sequence));
//                My_Assert(NBS.get_cur_cost() == initial_solution.total_cost && NBS.get_cur_vio_load() == initial_solution.total_vio_load,"Incorrect unpack!");
////                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                /*------------------------------consturct a start point-------------------------------*/
//
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                cout << "Begin Local search...\n";
//
//                /*------------------------------specify a start point-------------------------------*/
////                vector<int> test{-3,155,19,154,16,-60,65,174,117,177,130,185,184,148,145,142,-37,83,85,163,63,21,-17,38,42,164,66,-50,70,72,52,34,31,-20,39,64,162,159,-7,12,158,28,10,150,2,-41,160,49,90,91,69,43,68,-75,79,99,179,133,178,-80,96,97,170,98,95,54,-59,166,82,77,161,57,18,-9,30,44,114,126,122,121,140,136,180,134,132,-62,87,105,172,103,167,-5,23,27,29,33,35,47,26,-1,55,165,78,74,-51,73,93,169,94,118,71,45,-58,61,84,102,171,104,101,81,-86,111,124,143,144,175,125,112,-120,123,149,129,176,128,127,146,147,116,-6,157,22,156,4,-8,24,67,113,115,168,92,89,40,-88,109,173,110,107,-14,36,48,53,46,32,25,-11,152,13,153,15,151,-56,76,131,135,137,182,138,181,100,-106,141,183,139,119,108};
////                NBS.delimiter_coding_sol = get_delimeter_coding(test);
////                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
////                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                /*------------------------------specify a start point-------------------------------*/
//
//                struct timeb start_time;
//                ftime(&start_time);
//                for (int i = 1; i <= 100 ; i++) {
//                    double start_point = NBS.get_cur_cost();
//                    cout << "Starting Point: " << start_point << endl;
//                    struct timeb start_time;
//                    ftime(&start_time);
////                    SingleInsert::unit_test(NBS,Mixed_Instance);
////                    Invert::unit_test(NBS,Mixed_Instance);
////                    Slice::unit_test(NBS,Mixed_Instance);
////                    DoubleInsert::unit_test(NBS,Mixed_Instance);
////                    NewSwap::unit_test(NBS,Mixed_Instance);
////                    Extraction::unit_test(NBS,Mixed_Instance);
////                    NewTwoOpt::unit_test(NBS,Mixed_Instance);
//                    NBS.neighbor_search(Mixed_Instance);
////                    NBS.RTR_search(Mixed_Instance);
////                    NBS.infeasible_exploration(Mixed_Instance);
//                    struct timeb end_time;
//                    ftime(&end_time);
//                    cout << "Finish " << i << "th search, spent: "
//                         <<fixed << (end_time.time - start_time.time)
//                             + ((end_time.millitm - start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                }
//                struct timeb end_time;
//                ftime(&end_time);
//                cout << "Finish " << start_seed << "th search, spent: "
//                     <<fixed << (end_time.time - start_time.time)
//                         + ((end_time.millitm - start_time.millitm) * 1.0 / 1000) << 's' << endl;
//
////                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//            }
//            /*----------------------------------------------------------*/


//            /*----------------------------------------------------------*/
////            Local Search
//            {
//                /*------------------------------construct a start point-------------------------------*/
//                Individual initial_solution;
//                nearest_scanning(Mixed_Instance,initial_solution);
//                NBS.delimiter_coding_sol = initial_solution.sequence;
//                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                /*------------------------------consturct a start point-------------------------------*/
//
//
//                ftime(&epoch_start_time);
//                struct timeb cur_time;
//                cout << "Begin Local search...\n";
//
//                /*------------------------------specify a start point-------------------------------*/
////                vector<int> test{-3,155,19,154,16,-60,65,174,117,177,130,185,184,148,145,142,-37,83,85,163,63,21,-17,38,42,164,66,-50,70,72,52,34,31,-20,39,64,162,159,-7,12,158,28,10,150,2,-41,160,49,90,91,69,43,68,-75,79,99,179,133,178,-80,96,97,170,98,95,54,-59,166,82,77,161,57,18,-9,30,44,114,126,122,121,140,136,180,134,132,-62,87,105,172,103,167,-5,23,27,29,33,35,47,26,-1,55,165,78,74,-51,73,93,169,94,118,71,45,-58,61,84,102,171,104,101,81,-86,111,124,143,144,175,125,112,-120,123,149,129,176,128,127,146,147,116,-6,157,22,156,4,-8,24,67,113,115,168,92,89,40,-88,109,173,110,107,-14,36,48,53,46,32,25,-11,152,13,153,15,151,-56,76,131,135,137,182,138,181,100,-106,141,183,139,119,108};
////                NBS.delimiter_coding_sol = get_delimeter_coding(test);
////                NBS.unpack_seq(NBS.delimiter_coding_sol, Mixed_Instance);
////                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                /*------------------------------specify a start point-------------------------------*/
//
//                for (int i = 1; i <= 100 ; i++) {
//                    double start_point = NBS.cur_solution_cost;
//                    cout << "Starting Point: " << start_point << endl;
//                    struct timeb start_time;
//                    ftime(&start_time);
//                    NBS.neighbor_search(Mixed_Instance);
////                    NBS.RTR_search(Mixed_Instance);
////                    if(i % 5 == 0){
////                        NBS.infeasible_exploration(Mixed_Instance);
////                    }
//                    struct timeb end_time;
//                    ftime(&end_time);
//                    cout << "Finish " << i << "th search, spent: "
//                         <<fixed << (end_time.time - start_time.time)
//                             + ((end_time.millitm - start_time.millitm) * 1.0 / 1000) << 's' << endl;
//                }
//
//                Mixed_Instance.check_best_solution(NBS.cur_solution_cost, NBS.negative_coding_sol);
//                log_out.close();
//                cout << "Finished!\n\n\n";
//            }
//            /*----------------------------------------------------------*/


            /*----------------------------------------------------------*/
            //Memetic search
            {
                HighSpeedMemetic MA(NBS, pool_size, evolve_steps, QNDF_weights);
                cout << "Begin Memetic search...\n";
                for (int i = 1; i <= 10; i++) {
                    ftime(&epoch_start_time);
                    struct timeb cur_time;
                    MA.memetic_search(Mixed_Instance);
                    ftime(&cur_time);
                    cout << "Finish " << i << "th search, spent: "
                         << (cur_time.time - epoch_start_time.time)
                             + ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000) << 's' << endl;

                    MA.reset();
                }

                log_out.close();
                cout << "Finished!\n\n\n";
            }
            /*----------------------------------------------------------*/

            /* solution record */
            struct timeb cur_time;
            ftime(&cur_time);
            SearchtimeVec.push_back((cur_time.time - epoch_start_time.time) +
                ((cur_time.millitm - epoch_start_time.millitm) * 1.0 / 1000));

            /* record the best info of each epoch */
            fitnessVec.push_back(Mixed_Instance.best_total_route_length);
            BesttimeVec.push_back(Mixed_Instance.best_sol_time);

            cout << "Finish " << start_seed - random_seed + 1 << "th times\n";

            /* record the best solution during the whole searching process*/
            if (Mixed_Instance.best_total_route_length < bestobj) {
                best_solution_time = Mixed_Instance.best_sol_time;
                bestobj = Mixed_Instance.best_total_route_length;
                best_buffer = Mixed_Instance.best_sol_buff;
                best_epoch = start_seed - random_seed + 1;
            }

        }

        /***********************************output part********************************************/
        if (bestobj == numeric_limits<decltype(bestobj)>::max()) {
            cerr << "ERROR! Can't find a solution\n";
            abort();
        }
        double total_cost = sum(fitnessVec);
        double best_total_time = sum(BesttimeVec);
        double total_time = sum(SearchtimeVec);
        double average_cost = total_cost / fitnessVec.size();
        double average_best_time = best_total_time / BesttimeVec.size();
        double average_search_time = total_time / SearchtimeVec.size();

        cout << "\n\n**************************Searching Finish!**************************\n";
        cout << "Searching result:\n\n";


        result_out.open(result_dir + '/' + file_name + ".res", ios::out);
        result_out << setprecision(2) << fixed;

        print(cout, result_out, "Instance: " + file_name);
        print(cout, result_out, "Best Cost: " + to_string(bestobj));
        print(cout, result_out, "Best Cost Epoch: "+ to_string(best_epoch));
        print(cout, result_out, "Best Solution Search Time: " + to_string(best_solution_time) + 's');
        print(cout, result_out, "Best Solution: ");

        string buff = "";
        for (int i = 0; i < best_buffer.size(); i++) {
            if (i == best_buffer.size() - 1)
                buff += to_string(best_buffer[i]);
            else
                buff += (to_string(best_buffer[i]) + "->");
        }
        print(cout, result_out, buff);

        print(cout, result_out, "\n\nSearch time: " + to_string(fitnessVec.size()));


        print(cout, result_out, "\n\nAverage cost: " + to_string(average_cost));
        print(cout, result_out, "Average best solution search time: " + to_string(average_best_time) + 's');
        print(cout, result_out, "Average search time: " + to_string(average_search_time) + 's');
        result_out.close();
    }

    return 0;
}