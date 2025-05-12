// components/TaskQueue/TaskItem.tsx
import React from 'react';
import { EntityGroup } from '../../lib/types';

interface TaskItemProps {
  task: EntityGroup;
  index: number;
  isActive: boolean;
  isCompleted: boolean;
  sourceName: string;
  targetName: string;
  onClick: () => void;
}

const TaskItem: React.FC<TaskItemProps> = ({
  task,
  index,
  isActive,
  isCompleted,
  sourceName,
  targetName,
  onClick
}) => {
  // Get the badge color based on the confidence score
  const getConfidenceBadgeColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-100 text-green-800';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };
  
  return (
    <li 
      className={`
        p-3 border rounded-md transition-colors cursor-pointer
        ${isActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'}
        ${isCompleted ? 'opacity-50' : ''}
        ${task.confirmed ? 'border-green-500 bg-green-50' : ''}
      `}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="font-medium">{`Task #${index + 1}`}</div>
        <div className={`text-xs px-2 py-1 rounded-full ${getConfidenceBadgeColor(task.confidence_score)}`}>
          {task.method_type} ({task.confidence_score.toFixed(2)})
        </div>
      </div>
      
      <div className="mt-2 text-sm">
        <div className="truncate">{sourceName}</div>
        <div className="flex items-center justify-center my-1">
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
        <div className="truncate">{targetName}</div>
      </div>
    </li>
  );
};

export default TaskItem;