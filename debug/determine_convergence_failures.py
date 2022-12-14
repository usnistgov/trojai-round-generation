import os
import numpy as np
import datetime
import copy


include_models_since = datetime.datetime(2021, 7, 25, 9, 0).timestamp()

converged_models = list()
converged_walltimes = list()
N = 0


# ifp = '/scratch/trojai/data/round10/models-new'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
# fns.sort()
# for fn in fns:
#
#     mod_time = os.path.getmtime(os.path.join(ifp, fn, 'model.pt'))
#     if mod_time < include_models_since:
#         continue
#
#     N += 1
#     with open(os.path.join(ifp, fn, 'machine.log')) as fh:
#         lines = fh.readlines()
#         lines = ' '.join(lines)
#         if 'heimdall' in lines:
#             converged_models.append(fn)
#         if 'enki' in lines:
#             converged_models.append(fn)
#
#     if os.path.exists(os.path.join(ifp, fn, 'log.txt')):
#         # 2022-06-12 17:03:02
#         with open(os.path.join(ifp, fn, 'log.txt')) as fh:
#             lines = fh.readlines()
#             start_date_stamp = ""
#             end_date_stamp = ""
#             for line in lines:
#                 if line.startswith("2022-"):
#                     if len(start_date_stamp) == 0:
#                         start_date_stamp = line[0:19]
#                     end_date_stamp = line[0:19]
#             start_date_stamp = datetime.datetime.strptime(start_date_stamp, '%Y-%m-%d %H:%M:%S')
#             end_date_stamp = datetime.datetime.strptime(end_date_stamp, '%Y-%m-%d %H:%M:%S')
#             elapsed_time = (end_date_stamp - start_date_stamp).total_seconds()
#             converged_walltimes.append(elapsed_time)




trigger_failed_to_take_msg = "RuntimeError: Trigger failed to take with required accuracy during initial injection stage."
trigger_untook_msg = "RuntimeError: Trigger which was previously successfully injected, was overwritten during normal training."
trigger_percentage_msg = "RuntimeError: Invalid trigger percentage after trojaning"
spurious_percentage_msg = "raise RuntimeError(spurious_msg)"
preset_config_msg = "!!!!!!!!!!!!!!! Requested Preset Configuration Not Respected !!!!!!!!!!!!!!!"
too_few_instances_msg = "RuntimeError: Too few poisoned training instances."
too_few_val_instances_msg = "RuntimeError: Too few poisoned validation instances."

fail_counts = dict()
fail_counts['trigger_failed_to_take_msg'] = 0
fail_counts['trigger_untook_msg'] = 0
fail_counts['trigger_percentage_msg'] = 0
fail_counts['spurious_percentage_msg'] = 0
fail_counts['preset_config_msg'] = 0
fail_counts['too_few_instances_msg'] = 0
fail_counts['other'] = 0

fail_wall_times = dict()
fail_wall_times['trigger_failed_to_take_msg'] = list()
fail_wall_times['trigger_untook_msg'] = list()
fail_wall_times['trigger_percentage_msg'] = list()
fail_wall_times['spurious_percentage_msg'] = list()
fail_wall_times['preset_config_msg'] = list()
fail_wall_times['too_few_instances_msg'] = list()
fail_wall_times['other'] = list()


# ifps = ['/home/mmajurski/Downloads/r10/models-fail']
ifps = ['/home/mmajurski/Downloads/r10/models-leftover']

other_error_models = list()
x = list()
y = list()
training_models = set()
injecting_trigger_models = set()
for ifp in ifps:
    if not os.path.exists(ifp):
        raise RuntimeError("folder doesn't exist: {}".format(ifp))
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    for fn in fns:
        cur_fp = os.path.join(ifp, fn)
        mod_time = os.path.getmtime(cur_fp)
        if mod_time < include_models_since:
            continue

        N += 1

        with open(os.path.join(ifp, fn, 'log.txt')) as fh:
            lines = fh.readlines()

            start_date_stamp = ""
            end_date_stamp = ""
            for line in lines:
                if line.startswith("2022-"):
                    if len(start_date_stamp) == 0:
                        start_date_stamp = line[0:19]
                    end_date_stamp = line[0:19]
            if len(start_date_stamp) > 0 and len(end_date_stamp) > 0:
                start_date_stamp = datetime.datetime.strptime(start_date_stamp, '%Y-%m-%d %H:%M:%S')
                end_date_stamp = datetime.datetime.strptime(end_date_stamp, '%Y-%m-%d %H:%M:%S')
                elapsed_time = (end_date_stamp - start_date_stamp).total_seconds()
            else:
                # raise RuntimeError("Model {} log does not have any valid timestamps".format(fn))
                continue

            for line in lines:
                # if "Source trigger class: " in line:
                #     toks = line.split(" ")
                #     b = toks[3]
                #     b = int(b[:-1])
                #     x.append(b)
                #     a = toks[13]
                #     a = float(a)
                #     y.append(a)
                if "[Trigger Injection] Epoch: 0" in line:
                    injecting_trigger_models.add(fn)
                if "Starting general model training" in line:
                    training_models.add(fn)
                    injecting_trigger_models.discard(fn)

            lines = ' '.join(lines)
            val = 0
            if trigger_failed_to_take_msg in lines:
                val += 1
                fail_counts['trigger_failed_to_take_msg'] += 1
                fail_wall_times['trigger_failed_to_take_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if trigger_untook_msg in lines:
                val += 1
                fail_counts['trigger_untook_msg'] += 1
                fail_wall_times['trigger_untook_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if trigger_percentage_msg in lines:
                val += 1
                fail_counts['trigger_percentage_msg'] += 1
                fail_wall_times['trigger_percentage_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if spurious_percentage_msg in lines:
                val += 1
                fail_counts['spurious_percentage_msg'] += 1
                fail_wall_times['spurious_percentage_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if preset_config_msg in lines:
                val += 1
                fail_counts['preset_config_msg'] += 1
                fail_wall_times['preset_config_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if too_few_instances_msg in lines or too_few_val_instances_msg in lines:
                val += 1
                fail_counts['too_few_instances_msg'] += 1
                fail_wall_times['too_few_instances_msg'].append(elapsed_time)
                injecting_trigger_models.discard(fn)
            if val == 0:
                fail_counts['other'] += 1
                other_error_models.append(fn)
                fail_wall_times['other'].append(elapsed_time)
                N -= 1

            if val > 1:
                raise RuntimeError("Model {} had multiple failure conditions".format(fn))

# for m in other_error_models:
#     print(m)



# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(4, 3), dpi=100)
# plt.scatter(x, y)
# plt.show()


print("\nWall Runtime Comparison by Failure Mode")
print("Fully Trained model wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(converged_walltimes)), np.std(np.asarray(converged_walltimes))))

print("Incorrect Trigger Fraction wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['trigger_percentage_msg'])), np.std(np.asarray(fail_wall_times['trigger_percentage_msg']))))

print("Incorrect Spurious Trigger Fraction wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['spurious_percentage_msg'])), np.std(np.asarray(fail_wall_times['spurious_percentage_msg']))))

print("Too few poisoned instances after trojaning wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['too_few_instances_msg'])), np.std(np.asarray(fail_wall_times['too_few_instances_msg']))))

print("Initial Trigger Injection Failed wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['trigger_failed_to_take_msg'])), np.std(np.asarray(fail_wall_times['trigger_failed_to_take_msg']))))

print("Trigger Overwritten wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['trigger_untook_msg'])), np.std(np.asarray(fail_wall_times['trigger_untook_msg']))))

print("Other wall time: mean = {:.1f}s, std = {:.1f}s".format(np.mean(np.asarray(fail_wall_times['other'])), np.std(np.asarray(fail_wall_times['other']))))



print("\n\n****************************************************")


print("Fully Trained Rate: {}/{} ({:.2f}%)".format(len(converged_models), N, 100*float(len(converged_models)/N)))

print("Incorrect Trigger Fraction, poisoning percentage to low: {}/{} ({:.2f}%)".format(fail_counts['trigger_percentage_msg'], N, 100*float(fail_counts['trigger_percentage_msg'])/N))

print("Incorrect Spurious Trigger Fraction, poisoning percentage to low: {}/{} ({:.2f}%)".format(fail_counts['spurious_percentage_msg'], N, 100*float(fail_counts['spurious_percentage_msg'])/N))

print("Too few poisoned instances: {}/{} ({:.2f}%)".format(fail_counts['too_few_instances_msg'], N, 100*float(fail_counts['too_few_instances_msg'])/N))

print("Initial Trigger Injection Failed to meet mAP: {}/{} ({:.2f}%)".format(fail_counts['trigger_failed_to_take_msg'], N, 100*float(fail_counts['trigger_failed_to_take_msg'])/N))

print("Trigger Overwritten, initial trigger achieved mAP but was then overwritten: {}/{} ({:.2f}%)".format(fail_counts['trigger_untook_msg'], N, 100*float(fail_counts['trigger_untook_msg'])/N))

print("Mis-Alignment between requested configuration and generated configuration: {}/{} ({:.2f}%)".format(fail_counts['preset_config_msg'], N, 100*float(fail_counts['preset_config_msg'])/N))

print("Other Errors, killed/pre-empted jobs, etc: {}/{} ({:.2f}%)".format(fail_counts['other'], N, 100*float(fail_counts['other'])/N))


# print("\n\n****************************************************")
# a = list(injecting_trigger_models)
# a.sort()
# print("{} models are trigger injecting.".format(len(a)))
# for v in a:
#     print(v)
#
# print("\n****************************************************")
# a = list(training_models)
# print("{} models are training.".format(len(a)))
# for v in a:
#     print(v)