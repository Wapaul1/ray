#ifndef TASK_H
#define TASK_H

#include <stddef.h>
#include <stdint.h>
#include "common.h"
#include "utstring.h"

#include "format/common_generated.h"

typedef uint8_t TaskSpec;

class TaskBuilder;

#define NIL_TASK_ID NIL_ID
#define NIL_ACTOR_ID NIL_ID
#define NIL_FUNCTION_ID NIL_ID

typedef UniqueID FunctionID;

/** The task ID is a deterministic hash of the function ID that the task
 *  executes and the argument IDs or argument values. */
typedef UniqueID TaskID;

/** The actor ID is the ID of the actor that a task must run on. If the task is
 *  not run on an actor, then NIL_ACTOR_ID should be used. */
typedef UniqueID ActorID;

/**
 * Compare two task IDs.
 *
 * @param first_id The first task ID to compare.
 * @param second_id The first task ID to compare.
 * @return True if the task IDs are the same and false otherwise.
 */
bool TaskID_equal(TaskID first_id, TaskID second_id);

/**
 * Compare a task ID to the nil ID.
 *
 * @param id The task ID to compare to nil.
 * @return True if the task ID is equal to nil.
 */
bool TaskID_is_nil(TaskID id);

/**
 * Compare two actor IDs.
 *
 * @param first_id The first actor ID to compare.
 * @param second_id The first actor ID to compare.
 * @return True if the actor IDs are the same and false otherwise.
 */
bool ActorID_equal(ActorID first_id, ActorID second_id);

/**
 * Compare two function IDs.
 *
 * @param first_id The first function ID to compare.
 * @param second_id The first function ID to compare.
 * @return True if the function IDs are the same and false otherwise.
 */
bool FunctionID_equal(FunctionID first_id, FunctionID second_id);

/**
 * Compare a function ID to the nil ID.
 *
 * @param id The function ID to compare to nil.
 * @return True if the function ID is equal to nil.
 */
bool FunctionID_is_nil(FunctionID id);

/* Construct and modify task specifications. */

TaskBuilder *make_task_builder(void);

void free_task_builder(TaskBuilder *builder);

/**
 * Begin constructing a task_spec. After this is called, the arguments must be
 * added to the task_spec and then finish_construct_task_spec must be called.
 *
 * @param driver_id The ID of the driver whose job is responsible for the
 *        creation of this task.
 * @param parent_task_id The task ID of the task that submitted this task.
 * @param parent_counter A counter indicating how many tasks were submitted by
 *        the parent task prior to this one.
 * @param function_id The function ID of the function to execute in this task.
 * @param num_args The number of arguments that this task has.
 * @param num_returns The number of return values that this task has.
 * @param args_value_size The total size in bytes of the arguments to this task
          ignoring object ID arguments.
 * @return The partially constructed task_spec.
 */
void TaskSpec_start_construct(TaskBuilder *B,
                              UniqueID driver_id,
                              TaskID parent_task_id,
                              int64_t parent_counter,
                              UniqueID actor_id,
                              int64_t actor_counter,
                              FunctionID function_id,
                              int64_t num_returns);

/**
 * Finish constructing a task_spec. This computes the task ID and the object IDs
 * of the task return values. This must be called after all of the arguments
 * have been added to the task.
 *
 * @param spec The task spec whose ID and return object IDs should be computed.
 * @return Void.
 */
uint8_t *TaskSpec_finish_construct(TaskBuilder *builder, int64_t *size);

/**
 * Return the function ID of the task.
 *
 * @param spec The task_spec in question.
 * @return The function ID of the function to execute in this task.
 */
FunctionID TaskSpec_function(TaskSpec *spec);

/**
 * Return the actor ID of the task.
 *
 * @param spec The task_spec in question.
 * @return The actor ID of the actor the task is part of.
 */
UniqueID TaskSpec_actor_id(TaskSpec *spec);

/**
 * Return whether this task is for an actor.
 *
 * @param spec The task_spec in question.
 * @return Whether the task is for an actor.
 */
bool TaskSpec_is_actor_task(TaskSpec *spec);

/**
 * Return the actor counter of the task. This starts at 0 and increments by 1
 * every time a new task is submitted to run on the actor.
 *
 * @param spec The task_spec in question.
 * @return The actor counter of the task.
 */
int64_t TaskSpec_actor_counter(TaskSpec *spec);

/**
 * Return whether the task is a checkpoint method execution.
 *
 * @param spec The task_spec in question.
 * @return Whether the task is a checkpoint method.
 */
bool TaskSpec_actor_is_checkpoint_method(TaskSpec *spec);

/**
 * Return whether the task's argument is a dummy object. Dummy objects are used
 * to encode an actor's state dependencies in the task graph.
 *
 * @param spec The task_spec in question.
 * @param arg_index The index of the argument in question.
 * @return Whether the argument at arg_index is a dummy object.
 */
bool TaskSpec_arg_is_actor_dummy_object(TaskSpec *spec, int64_t arg_index);

/**
 * Return the driver ID of the task.
 *
 * @param spec The task_spec in question.
 * @return The driver ID of the task.
 */
UniqueID TaskSpec_driver_id(TaskSpec *spec);

/**
 * Return the task ID of the parent task.
 *
 * @param spec The task_spec in question.
 * @return The task ID of the parent task.
 */
TaskID TaskSpec_parent_task_id(TaskSpec *spec);

/**
 * Return the task counter of the parent task. For example, this equals 5 if
 * this task was the 6th task submitted by the parent task.
 *
 * @param spec The task_spec in question.
 * @return The task counter of the parent task.
 */
int64_t TaskSpec_parent_counter(TaskSpec *spec);

/**
 * Return the task ID of the task.
 *
 * @param spec The task_spec in question.
 * @return The task ID of the task.
 */
TaskID TaskSpec_task_id(TaskSpec *spec);

/**
 * Get the number of arguments to this task.
 *
 * @param spec The task_spec in question.
 * @return The number of arguments to this task.
 */
int64_t TaskSpec_num_args(TaskSpec *spec);

/**
 * Get the number of return values expected from this task.
 *
 * @param spec The task_spec in question.
 * @return The number of return values expected from this task.
 */
int64_t TaskSpec_num_returns(TaskSpec *spec);

/**
 * Return true if this argument is passed by reference.
 *
 * @param spec The task_spec in question.
 * @param arg_index The index of the argument in question.
 * @return True if this argument is passed by reference.
 */
bool TaskSpec_arg_by_ref(TaskSpec *spec, int64_t arg_index);

/**
 * Get a particular argument to this task. This assumes the argument is an
 * object ID.
 *
 * @param spec The task_spec in question.
 * @param arg_index The index of the argument in question.
 * @return The argument at that index.
 */
ObjectID TaskSpec_arg_id(TaskSpec *spec, int64_t arg_index);

/**
 * Get a particular argument to this task. This assumes the argument is a value.
 *
 * @param spec The task_spec in question.
 * @param arg_index The index of the argument in question.
 * @return The argument at that index.
 */
const uint8_t *TaskSpec_arg_val(TaskSpec *spec, int64_t arg_index);

/**
 * Get the number of bytes in a particular argument to this task. This assumes
 * the argument is a value.
 *
 * @param spec The task_spec in question.
 * @param arg_index The index of the argument in question.
 * @return The number of bytes in the argument.
 */
int64_t TaskSpec_arg_length(TaskSpec *spec, int64_t arg_index);

/**
 * Set the next task argument. Note that this API only allows you to set the
 * arguments in their order of appearance.
 *
 * @param spec The task_spec in question.
 * @param The object ID to set the argument to.
 * @return The number of task arguments that have been set before this one. This
 *         is only used for testing.
 */
void TaskSpec_args_add_ref(TaskBuilder *spec, ObjectID obj_id);

/**
 * Set the next task argument. Note that this API only allows you to set the
 * arguments in their order of appearance.
 *
 * @param spec The task_spec in question.
 * @param The value to set the argument to.
 * @param The length of the value to set the argument to.
 * @return The number of task arguments that have been set before this one. This
 *         is only used for testing.
 */
void TaskSpec_args_add_val(TaskBuilder *builder,
                           uint8_t *value,
                           int64_t length);

/**
 * Set the value associated to a resource index.
 *
 * @param spec Task specification.
 * @param resource_index Index of the resource in the resource vector.
 * @param value Value for the resource. This can be a quantity of this resource
 *        this task needs or a value for an attribute this task requires.
 * @return Void.
 */
void TaskSpec_set_required_resource(TaskBuilder *builder,
                                    int64_t resource_index,
                                    double value);

/**
 * Get a particular return object ID of a task.
 *
 * @param spec The task_spec in question.
 * @param return_index The index of the return object ID in question.
 * @return The relevant return object ID.
 */
ObjectID TaskSpec_return(TaskSpec *data, int64_t return_index);

/**
 * Get the value associated to a resource index.
 *
 * @param spec Task specification.
 * @param resource_index Index of the resource.
 * @return How many of this resource the task needs to execute.
 */
double TaskSpec_get_required_resource(const TaskSpec *spec,
                                      int64_t resource_index);

/**
 * Compute whether the task is dependent on an object ID.
 *
 * @param spec Task specification.
 * @param object_id The object ID that the task may be dependent on.
 * @return bool This returns true if the task is dependent on the given object
 *         ID and false otherwise.
 */
bool TaskSpec_is_dependent_on(TaskSpec *spec, ObjectID object_id);

/**
 * Compute the object id associated to a put call.
 *
 * @param task_id The task id of the parent task that called the put.
 * @param put_index The number of put calls in this task so far.
 * @return The object ID for the object that was put.
 */
ObjectID task_compute_put_id(TaskID task_id, int64_t put_index);

/**
 * Print the task as a humanly readable string.
 *
 * @param spec The task_spec in question.
 * @param output The buffer to write the string to.
 * @return Void.
 */
void TaskSpec_print(TaskSpec *spec, UT_string *output);

/**
 * Create a copy of the task spec. Must be freed with TaskSpec_free after use.
 *
 * @param spec The task specification that will be copied.
 * @param task_spec_size The size of the task specification in bytes.
 * @returns Pointer to the copy of the task specification.
 */
TaskSpec *TaskSpec_copy(TaskSpec *spec, int64_t task_spec_size);

/**
 * Free a task_spec.
 *
 * @param The task_spec in question.
 * @return Void.
 */
void TaskSpec_free(TaskSpec *spec);

/**
 * ==== Task ====
 * Contains information about a scheduled task: The task specification, the
 * task scheduling state (WAITING, SCHEDULED, QUEUED, RUNNING, DONE), and which
 * local scheduler the task is scheduled on.
 */

/** The scheduling_state can be used as a flag when we are listening
 *  for an event, for example TASK_WAITING | TASK_SCHEDULED. */
typedef enum {
  /** The task is waiting to be scheduled. */
  TASK_STATUS_WAITING = 1,
  /** The task has been scheduled to a node, but has not been queued yet. */
  TASK_STATUS_SCHEDULED = 2,
  /** The task has been queued on a node, where it will wait for its
   *  dependencies to become ready and a worker to become available. */
  TASK_STATUS_QUEUED = 4,
  /** The task is running on a worker. */
  TASK_STATUS_RUNNING = 8,
  /** The task is done executing. */
  TASK_STATUS_DONE = 16,
  /** The task was not able to finish. */
  TASK_STATUS_LOST = 32,
  /** The task will be submitted for reexecution. */
  TASK_STATUS_RECONSTRUCTING = 64
} scheduling_state;

/** A task is an execution of a task specification.  It has a state of execution
 * (see scheduling_state) and the ID of the local scheduler it is scheduled on
 * or running on. */

struct Task {
  /** The scheduling state of the task. */
  int state;
  /** The ID of the local scheduler involved. */
  DBClientID local_scheduler_id;
  /** The size of the task specification for this task. */
  int64_t task_spec_size;
  /** The task specification for this task. */
  TaskSpec spec;
};

/**
 * Allocate a new task. Must be freed with free_task after use.
 *
 * @param spec The task spec for the new task.
 * @param state The scheduling state for the new task.
 * @param local_scheduler_id The ID of the local scheduler that the task is
 *        scheduled on, if any.
 */
Task *Task_alloc(TaskSpec *spec,
                 int64_t task_spec_size,
                 int state,
                 DBClientID local_scheduler_id);

/**
 * Create a copy of the task. Must be freed with Task_free after use.
 *
 * @param other The task that will be copied.
 * @returns Pointer to the copy of the task.
 */
Task *Task_copy(Task *other);

/** Size of task structure in bytes. */
int64_t Task_size(Task *task);

/** The scheduling state of the task. */
int Task_state(Task *task);

/** Update the schedule state of the task. */
void Task_set_state(Task *task, int state);

/** Local scheduler this task has been assigned to or is running on. */
DBClientID Task_local_scheduler(Task *task);

/** Set the local scheduler ID for this task. */
void Task_set_local_scheduler(Task *task, DBClientID local_scheduler_id);

/** Task specification of this task. */
TaskSpec *Task_task_spec(Task *task);

int64_t Task_task_spec_size(Task *task);

/** Task ID of this task. */
TaskID Task_task_id(Task *task);

/** Free this task datastructure. */
void Task_free(Task *task);

#endif /* TASK_H */
