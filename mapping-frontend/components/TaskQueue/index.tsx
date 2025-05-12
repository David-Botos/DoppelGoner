// components/TaskQueue/TaskQueue-updated.tsx
import React from 'react';
import { EntityGroup, EntityDetails } from '../../lib/types';
import TaskItem from './TaskItem';

interface TaskQueueProps {
  tasks: EntityGroup[];
  currentTaskIndex: number;
  onSelectTask: (index: number) => void;
  entityDetailsCache: Record<string, EntityDetails>;
}

const TaskQueue: React.FC<TaskQueueProps> = ({
  tasks,
  currentTaskIndex,
  onSelectTask,
  entityDetailsCache
}) => {
  // Helper function to get entity name from cache if available
  const getEntityName = (entityId: string): string => {
    if (entityDetailsCache[entityId]) {
      return entityDetailsCache[entityId].organization_name;
    }
    return `Entity ${entityId.substring(0, 8)}...`; // Fallback
  };

  return (
    <div className="h-full overflow-auto">
      <div className="mb-4">
        <h3 className="text-lg font-medium">Tasks to Review: {tasks.length}</h3>
        <p className="text-sm text-gray-600">
          Completed: {currentTaskIndex} / {tasks.length}
        </p>
      </div>
      
      <ul className="space-y-2">
        {tasks.map((task, index) => (
          <TaskItem
            key={task.id}
            task={task}
            index={index}
            isActive={index === currentTaskIndex}
            isCompleted={index < currentTaskIndex}
            sourceName={getEntityName(task.source)}
            targetName={getEntityName(task.target)}
            onClick={() => onSelectTask(index)}
          />
        ))}
      </ul>
    </div>
  );
};

export default TaskQueue;